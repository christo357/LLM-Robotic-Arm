"""
    Main Script to run the model. 
    
"""

import os
import time
import threading
import queue
import json
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import TQC
from openai import OpenAI
from huggingface_hub import hf_hub_download 
from dotenv import load_dotenv

import my_envs
from wrappers import ActiveObjectWrapper, ManualGoalWrapper

load_dotenv()

HF_REPO_ID = "christo357/TQC_FetchPickAndPlace_v4"
MODEL_FILENAME = "tqcdense_model.zip"
VECNORM_FILENAME = "tqcdense_vecnorm.pkl"
# ==========================================
# 1. LLM Configuration
# ==========================================
# Option A: Groq 
API_KEY = os.getenv('GROK_API') # 
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant" 
# MODEL_NAME = "openai/gpt-oss-120b"

# Option B: Local Ollama
# API_KEY = "ollama"
# BASE_URL = "http://localhost:11434/v1"
# MODEL_NAME = "llama3.2"

# Option C: OpenAI GPT-4o
# API_KEY = "sk-..."
# BASE_URL = None 
# MODEL_NAME = "gpt-4o"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==========================================
# 2. The Semantic Parser 
# ==========================================
class LLMParser:
    def __init__(self):
        self.colors = ["black", "blue", "green", "yellow", "purple"]
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "robot_controller",
                    "description": "Control Fetch Robot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "enum": ["control", "scene", "focus", "goal", "sequence", "reset", "exit"],
                                "description": "focus=SELECT object. goal=MOVE object. control=start/stop physics."
                            },
                            "action": {
                                "type": "string", 
                                "enum": ["start", "stop"],
                                "description": "Only for intent='control'"
                            },
                            
                            # --- Goal Parameters ---
                            "target_color": {
                                "type": "string", 
                                "enum": self.colors,
                                "description": "Color to focus/select."
                            },
                            "relative_to": {
                                "type": "string", 
                                "enum": self.colors + ["table"],
                                "description": "Reference point. Use 'table' for absolute coordinates."
                            },
                            "position": {
                                "type": "string", 
                                "enum": ["above", "stack", "left", "right", "front", "center", "right_edge", "left_edge"],
                                "description": "Relative offset OR table location."
                            },
                            # --- Explicit Coordinates ---
                            "dest_x": {"type": "number", "description": "Absolute X [1.15 - 1.45]"},
                            "dest_y": {"type": "number", "description": "Absolute Y [0.60 - 1.00]"},
                            "dest_z": {"type": "number", "description": "Absolute Z [0.425 - 0.86]"},
                            
                            # --- Sequence/Scene ---
                            "n_objects": {
                                "type": "integer", 
                                "minimum": 1, 
                                "maximum": 5, 
                                "description": "Number of objects. Use with 'scene' OR 'reset'."
                            },
                            "steps": {
                                "type": "array",
                                "description": "List of commands for intent='sequence'",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "intent": {"type": "string"},
                                        "target_color": {"type": "string"},
                                        "relative_to": {"type": "string"},
                                        "position": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["intent"]
                    }
                }
            }
        ]

    def parse(self, user_text, current_n_objects):
        active_colors = self.colors[:current_n_objects]
        
        system_prompt = f"""
        You are a robot controller. You control a Fetch Robot on a table (height 0.42m).
        Active Objects: {', '.join(active_colors)}.
        COORDINATE SYSTEM (Meters):
        - X (Depth): 1.45 (Front/Far) to 1.15 (Back/Near). 
        - Y (Width): 0.60 (Right) to 1.00 (Left). 
        - Z (Height): 0.425 (Table Surface) to 0.87
        
        RULES:
        1. INTENT: 'GOAL' (Movement/Positioning)
            Trigger words: "Move", "Put", "Place", "Set target", "Lift", "Raise", "Above", "Next to".
            Rule: If the user mentions a location (e.g., "above green", "to the right"), it is ALWAYS 'goal'.
            
            Example 1. "Move yellow to right edge" -> intent: goal, target_color: "yellow",relative_to: "table", position: "right_edge".(Note: Must include 'target_color' so the robot knows WHAT to move).
            Example 2. "Lift green .2m" -> intent: goal, target_color: "green", relative_to: "table", dest_z: 0.625.
            Exmaple 3. "Lift blue 0.2m above table" -> intent: goal, target_color: "blue",relative_to: "table", dest_z: 0.625 (0.425+0.2).
            Example 4. "Set target above green" -> {{ "intent": "goal", "relative_to": "green", "position": "above" }}
                  (If user doesn't say WHAT to move, omit target_color, robot uses current).
        
        2. INTENT: 'FOCUS' (Selection Only)
           Trigger words: "Look at", "Select", "Switch to", "Focus on".
           Rule: NEVER use 'focus' if there is a spatial preposition (above, on, at).
           Example: "Select blue" -> {{ "intent": "focus", "target_color": "blue" }}.
        3. "Stack X on Y" -> Sequence: Focus X -> Goal Above X -> Goal Above Y -> Goal Stack Y.
        EXAMPLE: "Stack blue on red"
        Output:
        {{
          "intent": "sequence",
          "steps": [
             {{"intent": "focus", "target_color": "blue"}}, 
             {{"intent": "goal", "relative_to": "blue", "position": "above"}},  // Step 1: Lift Self
             {{"intent": "goal", "relative_to": "red", "position": "above"}},   // Step 2: Move Over Base
             {{"intent": "goal", "relative_to": "red", "position": "stack"}}    // Step 3: Lower/Place
          ]
        }}
        4. If the coordinates asked are not within the range of coordinate system, use the maximum/minimum possible value. 
        Eg. if the user asks to raise 1 m above table, set a target at height  0.87(maximum height), and random x, y coordinates.
        """

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                tools=self.tools,
                tool_choice="required", 
                temperature=0.0
            )

            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            args["valid"] = True
            return args

        except Exception as e:
            return {"valid": False, "reason": f"Tool Error: {str(e)}"}
        
# ==========================================
# 3. Global State & Threads
# ==========================================
command_queue = queue.Queue()
state = {
    "running": False,
    "n_objects": 3,
    "reset_req": False, 
    "exit_req": False,  
    "last_goal_cmd": None
}

def input_listener(llm_parser):
    print("\n🤖 AI Robot Interface Ready.Type 'exit' to quit. And 'reset' for default config")
    print("Try: 'Put the target above the blue block' or 'Use only 2 objects'")
    print("-" * 50)
    
    while True:
        try:
            user_text = input("\nUser Command >> ")
            if not user_text: continue
            
            print("Thinking...", end="\r")
            
            # Send to LLM with CURRENT state context
            cmd = llm_parser.parse(user_text, state["n_objects"])
            
            if cmd.get("valid") is False:
                print(f"❌ {cmd.get('reason', 'Invalid Request')}")
            else:
                command_queue.put(cmd)
                
        except EOFError:
            break

# ==========================================
# 4. Helpers
# ==========================================
def get_model_path(filename):
    """
    Checks local 'models/' folder. If missing, downloads from Hugging Face.
    """
    local_path = os.path.join("models", filename)
    
    if os.path.exists(local_path):
        return local_path
    
    print(f"⬇️ {filename} not found locally. Downloading from HF: {HF_REPO_ID}...")
    try:
        # Downloads to a cached location managed by HF
        cached_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        print(f"✅ Downloaded to: {cached_path}")
        return cached_path
    except Exception as e:
        print(f"❌ Error downloading from HF: {e}")
        return None
    
def create_env(n_objects):
    gym.register_envs(gymnasium_robotics)
    env = gym.make(
        'MultiObjectFetchPickAndPlace-v0',
        render_mode='human', 
        reward_type='dense', 
        n_objects=n_objects, 
        max_episode_steps=1000,
    )
    env = ActiveObjectWrapper(env, 0, n_objects)
    env = ManualGoalWrapper(env)
    
    venv = DummyVecEnv([lambda: env])
    try:
        venv = VecNormalize.load('models/tqcdense_vecnorm.pkl', venv)
        venv.training = False; venv.norm_reward = False
    except:
        print("⚠️ Warning: normalization file not found.")
    
    return venv

def apply_goal_logic(env, cmd, colors_map, state_n_objects):
    """
    1. Switches Focus (if target_color provided).
    2. Sets Goal (Relative or Absolute).
    Parses a goal command and updates the environment.
    Returns True if successful, False if target invalid.
    """
    
    # --- STEP 1: AUTO-FOCUS LOGIC ---
    if "target_color" in cmd and cmd["target_color"]:
        color = cmd["target_color"].lower()
        tid = colors_map.get(color)
        if tid is not None and tid < state_n_objects:
            print(f"👀 Auto-Switching Focus to {color} (ID: {tid})")
            env.env_method("set_active_object", tid)
        else:
            print(f"⚠️ Cannot focus: {color} not found.")
            return False

    # --- STEP 2: SET GOAL ---
    
    # ----------------------------------------
    # CASE A: Absolute Table Coordinates
    # ----------------------------------------
    if cmd.get("relative_to") == "table" or cmd.get("dest_x") is not None:
        # Default defaults (Center of table)
        x = cmd.get("dest_x", 1.30)
        y = cmd.get("dest_y", 0.80)
        z = cmd.get("dest_z", 0.43) # Just slightly above surface

        # Handle Semantic Table Positions (Overrides explicit coords if set)
        pos_name = cmd.get("position")
        if pos_name == "center":
            x, y = 1.30, 0.80
        elif pos_name == "right_edge":
            y = 0.60 # Near 0.38
        elif pos_name == "left_edge":
            y = 1.00 # Near 0.68
        elif pos_name == "front":
            x = 1.40 # Far from robot
        elif pos_name == "back":
            x = 1.15 # Near from robot

        # Safety Clamps (User provided bounds)
        x = np.clip(x, 1.15, 1.43)
        y = np.clip(y, 0.60, 1.00)
        z = np.clip(z, 0.425, 0.86)

        print(f"📍 Setting Absolute Goal: [{x:.2f}, {y:.2f}, {z:.2f}]")
        
        try:
            env.env_method("set_goal", np.array([x, y, z]))
            return True
        except Exception as e:
            print(f"⚠️ Error setting absolute goal: {e}")
            print("Make sure your Wrapper has a `set_goal(self, goal_pos)` method!")
            return False

    # ----------------------------------------
    # CASE B: Relative to Object (Old Logic)
    # ----------------------------------------
    
    ref_color = cmd["relative_to"].lower()
    tid = colors_map.get(ref_color)

    # Validation: Does this object exist in the current scene?
    if tid is None or tid >= state_n_objects:
        print(f"⚠️ Cannot set target: '{ref_color}' object not present.")
        return False

    pos_type = cmd["position"]
    
    # Calculate Offset
    offset = (0.0, 0.0, 0.0)
    if pos_type == "above":
        offset = (0.0, 0.0, 0.10) 
    elif pos_type == "stack":
        offset = (0.0, 0.0, 0.05)
    elif pos_type == "left":
        offset = (0.0, 0.10, 0.0)
    elif pos_type == "right":
        offset = (0.0, -0.10, 0.0)
    elif pos_type == "front":
        offset = (0.10, 0.0, 0.0)

    # Execute
    env.env_method("set_goal_relative_to_object", tid, offset)
    return True

# ==========================================
# 5. Main Execution Loop
# ==========================================
if __name__ == "__main__":
    # Initialize Logic
    llm = LLMParser()
    colors_map = {"black": 0, "blue": 1, "green": 2, "yellow": 3, "purple": 4}
    
    # Start Input Thread
    t = threading.Thread(target=input_listener, args=(llm,), daemon=True)
    t.start()
    
    
    # Load Env
    env = create_env(state["n_objects"])
    # 3. Load Model from HF
    model_path = get_model_path(MODEL_FILENAME)
    if not model_path:
        print("❌ CRITICAL: Model file missing. Exiting.")
        exit()
        
    model = TQC.load(model_path, env=env)
    obs = env.reset()
    
    void_action = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    print("Visualizer Started. Waiting for commands...")

    while not state["exit_req"]:
        # A. Process LLM Commands
        while not command_queue.empty():
            main_cmd = command_queue.get()
            
            # Check for Exit
            if main_cmd.get("intent") == "exit":
                state["exit_req"] = True
                break

            # Check for Manual Reset
            if main_cmd.get("intent") == "reset":
                state["n_objects"] = main_cmd.get("n_objects", 4)
                # state["last_goal_cmd"] = None # Clear targets
                state["reset_req"] = True
                break
            
            # Helper to wrap single commands into a list so we can treat everything as a sequence
            if main_cmd.get("intent") == "sequence":
                steps = main_cmd["steps"]
                print(f"📋 Received Sequence with {len(steps)} steps.")
                state['running'] = True
            else:
                steps = [main_cmd] # single command 
            
            # --- EXECUTE STEPS SEQUENTIALLY ---
            for i, cmd in enumerate(steps):
                if state["reset_req"] or state["exit_req"]: break # Safety break
                intent = cmd.get("intent")
                
                print(f"   👉 Step{i+1}: {intent} -> {cmd}")

                if intent == "control":
                    state["running"] = (cmd["action"] == "start")
                
                elif intent == "focus":
                    color = cmd["target_color"].lower()
                    tid = colors_map.get(color, 0)
                    if tid < state["n_objects"]:
                        env.env_method("set_active_object", tid)
                        obs, _, _, _ = env.step(void_action)     # Quick refresh

                    
                elif intent == "goal":
                    
                    success = apply_goal_logic(env, cmd, colors_map, state["n_objects"])
                    if success:
                        obs, _, _, _ = env.step(void_action)
                        
                        # print("      ... Executing movement ...")
                        # --- MOVEMENT LOOP ---
                        print("      ... Moving ...")
                        for _ in range(30): 
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, _, _ = env.step(action)
                            env.render()
                            time.sleep(0.01)
                        
                        # --- STABILIZATION (Fix Vibration) ---
                        print("      🛑 Stabilizing...")
                        stop_action = np.array([[0.0, 0.0, 0.0, action[0,3]]], dtype=np.float32)
                        for _ in range(10):
                            obs, _, _, _ = env.step(stop_action)
                            env.render()
                            time.sleep(0.01)
                            
                        # --- RELEASE LOGIC  ---
                        if cmd.get("position") == "stack":
                            print("      👐 Releasing block...")

                            # PHASE 1: RELEASE (Stop moving, Open Gripper)
                            # 0.0 for X,Y,Z to stop vibrations instantly.
                            # 1.0 for Gripper to force it open.
                            stop_action = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                            
                            for _ in range(15):
                                obs, _, _, _ = env.step(stop_action)
                                env.render()
                                time.sleep(0.01)

                            # PHASE 2: RETRACT (Move Up, Keep Open)
                            # Manually add +1.0 to Z to lift the arm straight up.
                            # This prevents the gripper from clipping the block when moving away.
                            lift_action = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
                            
                            for _ in range(10): # Lift for ~0.8 seconds
                                obs, _, _, _ = env.step(lift_action)
                                env.render()
                                time.sleep(0.01)
                                
                                
            if main_cmd.get("intent") in ["goal", "sequence"]:
                 print("✅ Move complete. Pausing physics.")
                 state["running"] = False
                                


        # B. Handle Reset
        if state["reset_req"]:
            print(f"--> Resetting Scene: {state['n_objects']} objects.")
            env.close()
            
            # Re-create environment
            env = create_env(state["n_objects"])
            model.set_env(env)
            obs = env.reset()
            
            state["reset_req"] = False
            state["running"] = False
            print(f"--> Scene Reset with {state['n_objects']} objects.")

        if state["running"]:
            # If "Start" was pressed, run the model indefinitely
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.01)
        else:
            # If "Stop" or idle, just render the scene
            env.render()
            time.sleep(0.1)
