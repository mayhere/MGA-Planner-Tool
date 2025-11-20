**# MGA Planner Tool**
Multiple Gravity Assist Interplanetary Trajectory Planner

MGA Planner Tool is a Python framework for designing and optimizing interplanetary trajectories using Multiple Gravity Assist (MGA) techniques.
It focuses on efficient Earth → Mercury transfers and naturally extends to other inner-planet MGA sequences.

Powered by Poliastro, NASA SPICE (SpiceyPy), and Differential Evolution, the tool automates low-fidelity MGA design and provides fast, flexible evaluation of candidate trajectories.

**✨ Key Features**
Lambert point-conics transfer solver for multi-leg MGA chains
V∞ matching to ensure flyby feasibility
Porkchop plots for visualizing launch/arrival energy
Launch window scanning (2025–2035)
C3L and bending-angle filtering
Tisserand consistency checks
Heliocentric trajectory plotting & MGA sequence visualization

This framework bridges analytical astrodynamics, numerical optimization, and trajectory visualization to support low-energy, high-efficiency mission design.

**📊 Example Plots (Included in Repo)**
1. Porkchop Plot (Earth → Mercury)
   Visualizes ΔV or C3 against launch/arrival dates.
   <img width="2100" height="2100" alt="2033-34_plot_c3l_tof" src="https://github.com/user-attachments/assets/faf95bfa-899d-410c-b39d-138b263c8c02" />

3. MGA Sequence Diagram
   Shows the chosen MGA chain (e.g., Earth → Venus → Venus → Mercury) with geometry markers.
   <img width="1124" height="635" alt="2026-27_EVVVMe" src="https://github.com/user-attachments/assets/0b1b08bb-2438-4499-98ce-78988486874d" />
  
3. Heliocentric Trajectory Plot
   Plots the Lambert arcs and planetary positions for the selected MGA route.
   <img width="1350" height="635" alt="2026-27_Earth-Mercury_Transfer" src="https://github.com/user-attachments/assets/d3cfbc2c-b16e-42af-8340-5bfe8372c6c6" />

**🧰 Tech Stack**
Python 3.10+
Poliastro
SpiceyPy / NASA SPICE kernels
NumPy, SciPy
Matplotlib
Differential Evolution optimizer

**📦 Installation**
git clone the repo
cd mga-planner-tool
pip install -r requirements.txt
Ensure you have SPICE kernels (e.g., de440.bsp) downloaded.

**🛰 Workflow Overview**
  <img width="3840" height="3302" alt="Mercury MGA Flow Chart" src="https://github.com/user-attachments/assets/3cefe3ac-47c1-4422-94c0-304bcc352641" />

**📄 License**
Released under the MIT License, fully compatible with Poliastro, SPICE, and other permissive dependencies.

**📚 Citation**
If you use this tool in research, kindly cite this repository.

**🤝 Contributing**
Issues and pull requests are welcome.

**📬 Contact**
Feel free to reach out for discussions related to orbital mechanics, gravity-assist planning, or mission design.
