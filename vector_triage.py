import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

# Dense Information Packing. No fluff. Maximum technical nouns.
portfolio = {
    "alrs_007_capsule": "C++ firmware dual-MCU architecture. I2C sensor data logging. IMU integration. Finite state machine for parachute descent control and telemetry payload.",
    "autonomous_terraforming_robot": "C++ kinematic control loops. Arduino GIGA R1 dual-core microcontroller. LiDAR obstacle avoidance and spatial mapping. Hardware PWM actuator control.",
    "aegis_swarm": "ROS 2 Jazzy distributed drone swarm autonomy. FastDDS middleware configuration. NVIDIA Isaac Sim digital twin simulation. Meta Quest 3 OpenXR teleoperation via Unity C#.",
    "comp0211_control_theory": "Mathematical control theory. Linear Quadratic Regulators (LQR), Model Predictive Control (MPC), Sliding Mode Control. Lyapunov stability analysis and Control Barrier Functions in Python.",
    "skimlit_pubmed": "Python TensorFlow deep learning. Natural language processing classification of biomedical RCT abstract sentences. Data preprocessing pipelines.",
    "c_terminal_game": "C programming memory management. Pointers, structs, and terminal I/O game loop.",
    "robotics_challenge": "Electronics hardware integration. Breadboard circuit wiring, microcontroller logic, CAD Autodesk Fusion."
}

projects = list(portfolio.keys())
enriched_descriptions = [
    f"{key.replace('_', ' ')}: {value}" 
    for key, value in portfolio.items()
]

project_embeddings = model.encode(enriched_descriptions)

# The Test
job_query = "Seeking robotics engineer with ROS 2, simulation experience, and C++ control loops."
query_embedding = model.encode([job_query])

scores = cosine_similarity(query_embedding, project_embeddings)[0]

# Display all scores to understand the metric space
print(f"Target Job: {job_query}\n")
for i, score in enumerate(scores):
    print(f"[{score:.3f}] {projects[i]}")

best_idx = np.argmax(scores)
best_score = scores[best_idx]
best_project = projects[best_idx]

print(f"\nTop Match: {best_project} ({best_score:.3f})")

# Empirically calibrated threshold for MiniLM
if best_score >= 0.40:
    print("Triage Passed: Routing to LLM Drafter...")
else:
    print("Triage Failed: Score below 0.40. Dropping application.")