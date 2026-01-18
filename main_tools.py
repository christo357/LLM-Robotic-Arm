"""
    Script to use LLM with tools for query
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
from dotenv import load_dotenv

load_dotenv()

# --- Custom Imports ---
import my_envs
from wrappers import ActiveObjectWrapper, ManualGoalWrapper

# ==========================================
# 1. LLM Configuration (EDIT THIS)
# ==========================================
# Option A: Groq (Fastest/Free)
API_KEY = os.getenv('GROK_API') # <--- Paste your Groq Key here
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant" 
# MODEL_NAME = "openai/gpt-oss-120b"

# Option B: Local Ollama
# API_KEY = "ollama"
# BASE_URL = "http://localhost:11434/v1"
# MODEL_NAME = "llama3.2"

# Option C: OpenAI GPT-4o
# API_KEY = "sk-..."
# BASE_URL = None # Defaults to OpenAI
# MODEL_NAME = "gpt-4o"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==========================================
# 2. The Semantic Parser (Tool Calling Edition)
# ==========================================
class LLMParser:
    def __init__(self):
        self.colors = ["black", "blue", "green", "yellow", "purple"]
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "robot_controller",
                    "description": "Control Fetch Robot. DISTINGUISH CLEARLY: 'Focus' = Select Object. 'Goal' = Move Object.",
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
        
        # We clarify the difference in the system prompt too
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
            
            Example 1. "Move blue to right edge" -> intent: goal, relative_to: "table", position: "right_edge".
            Example 2. "Move blue to x=1.3, y=0.4" -> intent: goal, relative_to: "table", dest_x: 1.3, dest_y: 0.4.
            Exmaple 3. "Lift blue 0.2m above table" -> intent: goal, relative_to: "table", dest_z: 0.625 (0.425+0.2).
            Example 4. "Set target above green" -> {{ "intent": "goal", "relative_to": "green", "position": "above" }}
        
        2. INTENT: 'FOCUS' (Selection Only)
           Trigger words: "Look at", "Select", "Switch to", "Focus on".
           Rule: NEVER use 'focus' if there is a spatial preposition (above, on, at).
           Example: "Select blue" -> {{ "intent": "focus", "target_color": "blue" }}.
        3. "Stack X on Y" -> Sequence: Focus X -> Goal Above X -> Goal Above Y -> Goal Stack Y.
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
# 4. Helper: Environment Setup
# ==========================================
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
    Parses a goal command and updates the environment.
    Returns True if successful, False if target invalid.
    """
    # ----------------------------------------
    # CASE A: Absolute Table Coordinates
    # ----------------------------------------
    if cmd.get("relative_to") == "table" or cmd.get("dest_x") is not None:
        # Default defaults (Center of table)
        x = cmd.get("dest_x", 1.30)
        y = cmd.get("dest_y", 0.53)
        z = cmd.get("dest_z", 0.43) # Just slightly above surface

        # Handle Semantic Table Positions (Overrides explicit coords if set)
        pos_name = cmd.get("position")
        if pos_name == "center":
            x, y = 1.30, 0.53
        elif pos_name == "right_edge":
            y = 0.40 # Near 0.38
        elif pos_name == "left_edge":
            y = 0.66 # Near 0.68
        elif pos_name == "front":
            x = 1.15 # Near robot
        elif pos_name == "back":
            x = 1.38 # Far from robot

        # Safety Clamps (User provided bounds)
        x = np.clip(x, 1.10, 1.40)
        y = np.clip(y, 0.38, 0.68)
        z = np.clip(z, 0.425, 0.86)

        print(f"📍 Setting Absolute Goal: [{x:.2f}, {y:.2f}, {z:.2f}]")
        
        # We assume your ManualGoalWrapper allows setting the goal directly.
        # If your wrapper doesn't have a specific method, we can usually
        # call the internal method or set the attribute if accessible.
        # Here we use a generic 'set_goal' method call.
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
    model = TQC.load('models/tqcdense_model.zip', env=env)
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
                if main_cmd.get("n_objects"):
                    state["n_objects"] = main_cmd["n_objects"]
                else:
                    state["n_objects"] = 4
                state["last_goal_cmd"] = None # Clear targets
                state["reset_req"] = True
                break
            
            # Helper to wrap single commands into a list so we can treat everything as a sequence
            if main_cmd.get("intent") == "sequence":
                steps = main_cmd["steps"]
                print(f"📋 Received Sequence with {len(steps)} steps.")
                state['running'] = True
            else:
                steps = [main_cmd] # Treat single command as a list of 1
            
            # --- EXECUTE STEPS SEQUENTIALLY ---
            for i, cmd in enumerate(steps):
                if state["reset_req"] or state["exit_req"]: break # Safety break
                intent = cmd.get("intent")
                
                print(f"   👉 Step{i+1}: {intent} -> {cmd}")

                if intent == "control":
                    state["running"] = (cmd["action"] == "start")
                
                elif intent == "scene":
                    new_n = cmd["n_objects"]
                    # Safety check for min/max objects
                    if new_n < 1: new_n = 1
                    if new_n > 5: new_n = 5
                    
                    state["n_objects"] = new_n
                    state["reset_req"] = True
                
                elif intent == "focus":
                    color = cmd["target_color"].lower()
                    tid = colors_map.get(color, 0)
                    if tid < state["n_objects"]:
                        env.env_method("set_active_object", tid)
                        # Quick refresh
                        obs, _, _, _ = env.step(void_action)
                    
                elif intent == "goal":
                    
                    success = apply_goal_logic(env, cmd, colors_map, state["n_objects"])
                    if success:
                        if intent != "sequence": # Only save persistent goal if not a temp sequence step
                             state["last_goal_cmd"] = cmd
                        obs, _, _, _ = env.step(void_action)
                        
                    if state["running"]:
                        print("      ... Executing movement ...")
                        # Wait 2.5 seconds (adjust based on your robot speed)
                        # We use a loop here to keep rendering while waiting
                        for _ in range(60): 
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, _, _ = env.step(action)
                            env.render()
                            time.sleep(0.04)
                            
                        # --- RELEASE LOGIC (New!) ---
                        # If this step was a "stack" operation, force the gripper open
                        if cmd.get("position") == "stack":
                            print("      👐 Releasing block...")
                            # Run for 20 frames (~1 sec) forcing gripper open
                            # action, _ = model.predict(obs, deterministic=True)

                            # PHASE 1: RELEASE (Stop moving, Open Gripper)
                            # We send 0.0 for X,Y,Z to stop vibrations instantly.
                            # We send 1.0 for Gripper to force it open.
                            stop_action = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                            
                            for _ in range(15):  # Hold for ~0.6 seconds
                                obs, _, _, _ = env.step(stop_action)
                                env.render()
                                time.sleep(0.04)

                            # PHASE 2: RETRACT (Move Up, Keep Open)
                            # We manually add +1.0 to Z to lift the arm straight up.
                            # This prevents the gripper from clipping the block when moving away.
                            lift_action = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
                            
                            for _ in range(10): # Lift for ~0.8 seconds
                                obs, _, _, _ = env.step(lift_action)
                                env.render()
                                time.sleep(0.04)
                                
            # --- SEQUENCE FINISHED ---
            # Stop the robot so it doesn't loop or drift.
            if main_cmd.get("intent") == "sequence":
                print("✅ Sequence Complete. Stopping.")
                state["running"] = False

        # B. Handle Reset
        if state["reset_req"]:
            print(f"--> Resetting Scene: {state['n_objects']} objects.")
            env.close()
            
            # Re-create environment
            env = create_env(state["n_objects"])
            model.set_env(env)
            obs = env.reset()
            
            # RE-APPLY PERSISTENT GOAL
            # If we had a target set (e.g., "above green"), re-apply it now
            if state["last_goal_cmd"]:
                print(f"--> Restoring target: {state['last_goal_cmd']['relative_to']}")
                apply_goal_logic(env, state["last_goal_cmd"], colors_map, state["n_objects"])
                # Refresh obs after setting goal
                obs, _, _, _ = env.step(void_action)
            
            state["reset_req"] = False
            state["running"] = False
            print(f"--> Scene Reset with {state['n_objects']} objects.")

        # C. Physics Step
        if state["running"]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.04)
        else:
            env.render()
            time.sleep(0.1)
    