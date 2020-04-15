MTF-CRNN: Multi-scale Time-Frequency Convolutional Recurrent Neural Network For Sound Event Detection
 based on DCASE2017 task2
 references:
Toni Heittola	Baseline system, DCASE Framework, Documentation	toni.heittola@tut.fi, http://www.cs.tut.fi/~heittolt/, https://github.com/toni-heittola
Aleksandr Diment	Dataset synthesis (Task 2)	aleksandr.diment@tut.fi, http://www.cs.tut.fi/~diment/
Annamaria Mesaros	Documentation	annamaria.mesaros@tut.fi, http://www.cs.tut.fi/~mesaros/
Documentation
See https://tut-arg.github.io/DCASE2017-baseline-system/ for detailed instruction, manuals and tutorials.

Getting started
Clone repository from Github or download latest release.
Install requirements with command: pip install -r requirements.txt
Run the application with default settings: python applications/task2.py
System description
This is the Multi-scale Time-Frequency Convolutional Recurrent Neural Network For Sound Event Detection for the Detection and Classification of Acoustic Scenes and Events 2017 (DCASE2017) challenge task 2.

The code is based on the baseline system.  we propose a multi-scale time-frequency convolutional recurrent neural network (MTF-CRNN) for sound event detection. We exploit four groups of parallel and serial convolutional kernels to learn high-level shift invariant features from the time and frequency domains of acoustic samples. A two-layer bi-directional gated recurrent unit is used  to capture the temporal context from the extracted high-level features. The proposed method is evaluated on two different sound event datasets. Compared to baseline method and other methods, the performance is greatly improved as a single model with few parameters without pre-training. On the TUT Rare Sound Events 2017 evaluation dataset, our method achieved an error rate(ER) of 0.09$\pm$0.01 which got an improvement of 83${\%}$ than the baseline. On the TAU Spatial Sound Events 2019 evaluation dataset, our system reports an ER of 0.11$\pm$0.01, a relative improvement over the baseline of 61${\%}$, and the F1 and ER is better than that of on the development dataset. Compared to the state-of-the-art methods, our proposed network achieves very competitive detection performance with few parameters and good generalization capability.

The main approach implemented in the system:

Acoustic features: Log Mel-band energies extracted in 40ms windows with 20ms hop size.
Machine learning: neural network approach using multi-scale time-frequency convolutional recurrent neural network (MTF-CRNN) for sound event detection (with 300 neurons each, and 20% dropout between layers).
Directory layout

.
├── applications            # Task specific applications (task2.py) 
│   └── parameters          # Default parameters for the applications
├── dcase_framework         # DCASE Framework code
│   └── application_core.py          # The main body for the applications
│   └── pytorch_utils.py         # The model code
├── README.md               # This file
└── requirements.txt        # External module dependencies 

Installation
The system is developed for Python 3.6. 5. This system is tested to work with Linux operating systems.

To get started, run command:

python3 task2.py

See more detailed instructions from documentation.
references:
MTF-CRNN:Multi-scale Time-Frequency Convolutional Recurrent Neural Network For Sound Event Detection
License
The DCASE Framework and the baseline system is released only for academic research under EULA.pdf from Tampere University of Technology.
