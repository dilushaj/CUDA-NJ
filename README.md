# CUDA-NJ
GPU accllerated program for Neighbor-Joining algorithm of Biopython library. The project includes both Serial Vesion and the GPU version of Neighbour Joining Algorithm. I have used PyCUDA API which maps CUDA into Python for the implementation convienience.  

## Getting Started

Clone or download the .zip file of the project and locate to the root directory.

### Prerequisites

Inorder to run the program you shold have following hardware and software requirements.
#### Software Requirements
```
Python 3
Biopython library install
PyCUDA
```
#### Hardware Requirements
```
GPU server
```
### Installation

Python 3  - https://www.python.org/downloads/
<br/><br/>
Biopython - https://biopython.org/wiki/Download
<br/><br/>
PyCUDA    - https://wiki.tiker.net/PyCuda/Installation

After installing above softwares you are good to go.

## Running the Program
To run the Serial Version of NJ algorithm run the follwing commands from root directory.
```
cd Serial-Version
python phylo_treeSerial.py
```
To run the GPU version of NJ algorithm run the following commands from root directory. 
```
cd GPU-Version
python phylo_treeGPU.py
```
## Testing for diffrent datasets

Test datasets(both Amino Acids and DNA) have been given in Datasets directory. Change the phylo_treeGPU.py file accordingly and test the results for different datasets.
