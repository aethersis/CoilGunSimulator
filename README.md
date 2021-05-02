# CoilGunSimulator
A set of utilities designed to simulate various coil gun parameters and optimize the designs. Based on 5 different real-world designs, the end velocity calculation error was always &lt;10%. Unfortunately it runs only on Windows since it uses LTSPICE and FEMM. 

## Installation
Windows 7 or newer is required. Tested with Python 3.7. 
1. Install LtSpice on your computer from https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html
2. Install FEMM on your computer from https://www.femm.info/wiki/Download
3. Install requirements.txt `pip3 install -r requirements.txt`

## Running
For now only a demo script is provided. Try running coilgun_simulator.py script.