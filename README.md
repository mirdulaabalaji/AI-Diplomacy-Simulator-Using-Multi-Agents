#  AI Diplomacy Simulator Using Multi-Agents

> *A four-agent, LLM-powered international negotiation simulator that combines natural language diplomacy, game-theoretic decision-making, exclusive coalition management, and cross-simulation persistent memory, built as a unified framework.*

---

##  Table of Contents

- [Motivation](#-motivation)
- [System Overview](#-system-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [License](#-license)
---

##  Motivation

Real-world diplomacy is not a mathematical puzzle: it involves **memory, trust, deception, and shifting alliances**. Yet no existing computational system adequately models all of these at once. 

- **Human-led negotiations** are slow, biased, and impossible to simulate at scale.
- **Classical multi-agent systems (MAS)** handle structured tasks but cannot operate in open-ended, dynamic diplomatic domains.
- **Game-theoretic models** provide equilibrium analysis but produce no natural language interaction or adaptive communication.
- **LLM-only systems** enable reasoning in natural language but lack strategic grounding and long-term consistency, making them prone to hallucinations and erratic defection.
- **Diplomacy-specific AI** as an integrated framework is virtually non-existent.

The cost of failed negotiations in the real world involves broken treaties, economic sanctions, armed conflict which makes AI-assisted simulation **extremely valuable** for policy analysis, training, and education. This project addresses that gap by building the first unified framework that combines **strategic reasoning, natural language negotiation, coalition dynamics, and long-term memory** in a single multi-agent system.

---

##  System Overview

The simulator models a **four-country international negotiation** as a repeated strategic game.  Each country is an **autonomous LLM-powered agent** that:

1. Sends private diplomatic messages to all other agents before deciding
2. Reads its inbox and applies game-theoretic reasoning to choose **Cooperate (C)** or **Defect (D)**
3. Proposes a bilateral coalition partner
4. Has its payoffs, trust scores, and memory updated after each round

The simulation ends when **all four agents cooperate simultaneously** (agreement reached) or the **maximum round cap** is hit (negotiation collapsed).

---

##  Features

- **4-Agent LLM Simulation**: each country is a fully autonomous strategic agent powered by Groq LLMs
-  **Two Distinct LLM Models**: behavioural diversity achieved by assigning different Groq models across agents
-  **Private Diplomatic Messaging**: 12 bilateral messages per round, context-aware, memory-informed
-  **Game-Theoretic Decision Making**: Tit-for-Tat, Nash Bargaining, Brinkmanship, Repeated Game, Deterrence Theory, Free-Rider Problem, Prisoner's Dilemma, and more
-  **LLM-Based Game Classifier**: dynamically identifies applicable game theory models per scenario
-  **Exclusive Coalition Management**: mutual consent, one-ally-per-country constraint, full lifecycle management
-  **Per-Opponent Memory**: cooperation rates, betrayal counts, alliance counts per agent pair
-  **Cross-Simulation Persistent Memory**: SQLite stores historical reputation data; injected into LLM prompts as "long-term reputation"
-  **Real-Time Streaming UI**: Streamlit with thread-safe event queue, monkey-patched simulation bridge
-  **Analytics Dashboard**: outcome distribution, agent leaderboard, trust timeline, coalition vs betrayal charts, game theory model frequency
-  **Guaranteed Termination**: conditional LangGraph edge ensures finite execution

---

##  Technologies Used

### Core Language & Framework
| Technology | Purpose |
|---|---|
| **Python** | Primary language: simulation control, agent logic, orchestration |
| **LangChain** | LLM invocation, prompt templating, ChatGroq integration |
| **LangGraph** | Graph-based agent orchestration, state management, conditional routing |
| **Pydantic** | Strict data validation: `NegotiationState`, `CountryState`, `CountryMemory`, `Message` |

### LLM & API
| Technology | Purpose |
|---|---|
| **Groq API** | Ultra-fast LLM inference |
| **LLaMA 3.1-8b-instant** | Primary reasoning model for structured decisions |
| **LLaMA 3.1-70b-versatile** | Secondary model for behavioural diversity across agents |

### Game Theory
| Module | Purpose |
|---|---|
| `game_theory/payoff_matrices.py` | Extended Prisoner's Dilemma matrix for 4-agent pairwise evaluation |
| `game_theory/equilibrium.py` | Brinkmanship risk computation, Nash Bargaining solution |
| **NumPy** | Trust score clamping, payoff calculations |

### Memory & Storage
| Technology | Purpose |
|---|---|
| **SQLite3** | Cross-simulation persistent memory (`negotiation_memory.db`) |
| `memory/long_term_memory.py` | Read/write simulation stats and agent reputation |
| **JSON** | Final structured simulation output (`outcomes.json`) |

### Visualization & UI
| Technology | Purpose |
|---|---|
| **Streamlit** | Real-time web UI |
| `utils/stream_bridge.py` | Thread-safe `queue.Queue` connecting simulation to UI |
| **Monkey-patching** | Runtime wrapping of core functions to push `SimEvent` objects |

---

##  Results

### Cooperation Scenario (Vaccine Distribution)
- Messaging phase built trust across 5 rounds
- All four agents converged to **Cooperate** - `agreement_reached = True`
- Game theory models applied: **Nash Bargaining + Repeated Game Equilibrium**
- Coalition bonuses applied; no free-rider penalties triggered
<img width="1101" height="563" alt="Screenshot 2026-03-22 192909" src="https://github.com/user-attachments/assets/0e4d8b25-be84-43df-93a1-2701277902e5" />


### Collapse Scenario (Aggressive Dominance)
- Country A (highly self-serving) persistently defected
- Trust collapsed: `cooperation_rate ↓`, `betrayal_count ↑`
- Free-rider penalties triggered; negotiation failed at `MAX_ROUNDS`
- Game theory outcome: **Prisoner's Dilemma trap + Brinkmanship escalation**
<img width="1365" height="601" alt="Screenshot 2026-03-22 195408" src="https://github.com/user-attachments/assets/de8bab6c-cc18-4dd6-9f20-c8824aee092c" />


### Analytics Dashboard (4 Simulations)
| Metric | Value |
|---|---|
| Total simulations | 4 |
| Full agreements | 2 (50%) |
| Collapsed | 2 (50%) |
| Average rounds | 3.5 |
| Most frequent game model | Repeated Game / Nash Bargaining |
<img width="1015" height="407" alt="Screenshot 2026-04-05 170654" src="https://github.com/user-attachments/assets/c78f8370-9989-40d0-9d25-8767814470d9" />
<img width="1296" height="557" alt="Screenshot 2026-04-05 170552" src="https://github.com/user-attachments/assets/a3120577-7a20-4e15-9cb0-988a950909ba" />


---

##  Getting Started

### Prerequisites
```bash
python >= 3.10
```

### Installation
```bash
git clone https://github.com/yourusername/ai-diplomacy-simulator.git
cd ai-diplomacy-simulator
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run the Simulator
```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser. Enter a negotiation scenario in natural language and click **Run Simulation**.

### Example Scenario Inputs
```
Country A is an oil superpower seeking export profits. Country B is an 
industrialized economy dependent on energy imports. Country C is a 
renewable energy leader pushing for green transition. Country D is a 
developing nation seeking affordable energy access.

Country A is aggressive and seeks cyber dominance. Country B is cautious 
and cooperative but fears betrayal. Country C promotes alliances. 
Country D is a regulatory body with limited enforcement power.
```
---

##  License

This project was developed as academic coursework at Vellore Institute of Technology. Please contact the author before reuse or redistribution.

---
