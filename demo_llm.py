"""
    Script to use LLM for query
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
# 2. The Semantic Parser (LLM)
# ==========================================
class LLMParser:
    def __init__(self):
        self.colors = ["black", "blue", "green", "yellow", "purple"]
        
    def get_system_prompt(self, current_n_objects):
        """
        Dynamically builds the context based on current scene state.
        """
        active_colors = self.colors[:current_n_objects]
        
        return f"""
        You are a robot controller. You control a Fetch Robot on a table (height 0.42m).
        
        CURRENT SCENE STATE:
        - Active Objects: {', '.join(active_colors)}
        - Object IDs: Red=0, Blue=1, Green=2, Yellow=3, Purple=4
        
        YOUR JOB:
        Parse the user's natural language into a JSON command.
        
        VALID INTENTS (JSON Schema):
        1. CONTROL: {{"intent": "control", "action": "start" | "stop"}}
        2. CHANGE SCENE: {{"intent": "scene", "n_objects": int (1-5)}}
        3. SET FOCUS: {{"intent": "focus", "target_color": string}}
           - VALIDATE: Is this color currently on the table? If not, return valid=false.
        4. SET GOAL: {{"intent": "goal", "relative_to": string, "position": "above" | "left" | "right" | "front"}}
           - "above" means on top (z ~ 0.55).
           - VALIDATE: Is the reference object valid?
        5. SEQUENCE (For complex tasks like 'stack')
        {{"intent": "sequence", "steps": [list of commands]}}
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
        6. RESET: {{"intent": "reset"}} (Resets env to default state)
        7. EXIT: {{"intent": "exit"}} (Stops the program)
        
        For "position":
        - "above" = z + 0.15 (Hover height)
        - "stack" = z + 0.05 (Place height)
        
           
        OUTPUT FORMAT:
        If valid: {{"valid": true, "intent": "...", ...params...}}
        If invalid/impossible: {{"valid": false, "reason": "Explain why"}}
        
        IMPORTANT: Return ONLY raw JSON. No markdown formatting.
        """
    
    

    def parse(self, user_text, current_n_objects):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.get_system_prompt(current_n_objects)},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            # Clean generic markdown if model adds it (```json ... ```)
            if content.startswith("```"):
                content = content.split("```")[1].strip()
                if content.startswith("json"): content = content[4:].strip()
            
            return json.loads(content)
            
        except Exception as e:
            return {"valid": False, "reason": f"LLM Error: {str(e)}"}

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
        max_episode_steps=1000
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
                state["n_objects"] = 4  # Restore default count
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
                    # break
                
                elif intent == "focus":
                    color = cmd["target_color"].lower()
                    tid = colors_map.get(color, 0)
                    if tid < state["n_objects"]:
                        env.env_method("set_active_object", tid)
                        # Quick refresh
                        obs, _, _, _ = env.step(void_action)
                    # env.env_method("set_active_object", tid)
                    # # Auto-set goal nearby so user sees the change
                    # # env.env_method("set_goal_relative_to_object", tid, (0, 0, 0.05))
                    # obs, _, _, _ = env.step(np.array([[0,0,0,0]]))
                    
                elif intent == "goal":
                    # # Logic for "Above" calculation
                    # ref_color = cmd["relative_to"].lower()
                    # tid = colors_map.get(ref_color, 0)
                    # pos_type = cmd["position"]
                    
                    # # Table is 0.42. Object center is ~0.445 (0.42 + 0.025). 
                    # # "Above" needs to clear the object height. 
                    # # If we want the *Goal* (red dot) to be at 0.55 relative to world Z?
                    # # The wrapper method 'set_goal_relative_to_object' adds offset to object position.
                    
                    # offset = (0.0, 0.0, 0.0)
                    # if pos_type == "above":
                    #     # Object is at Z ~0.42 (table). We want Z ~0.50.
                    #     # Offset = 0.50 - 0.42 = +0.08 roughly
                    #     offset = (0.0, 0.0, 0.10) 
                    # elif pos_type == "left":
                    #     offset = (0.0, 0.10, 0.0)
                    # elif pos_type == "right":
                    #     offset = (0.0, -0.10, 0.0)
                    # elif pos_type == "front":
                    #     offset = (0.10, 0.0, 0.0)

                    # env.env_method("set_goal_relative_to_object", tid, offset)
                    # obs, _, _, _ = env.step(np.array([[0,0,0,0]]))
                    
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
            # # If we had a target set (e.g., "above green"), re-apply it now
            # if state["last_goal_cmd"]:
            #     print(f"--> Restoring target: {state['last_goal_cmd']['relative_to']}")
            #     apply_goal_logic(env, state["last_goal_cmd"], colors_map, state["n_objects"])
            #     # Refresh obs after setting goal
            #     obs, _, _, _ = env.step(void_action)
            
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
    