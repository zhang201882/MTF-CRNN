# MTF-CRNN
Inspired by the convolutional recurrent neural network(CRNN) and inception, we propose a multiscale time-frequency convolutional recurrent
neural network (MTF-CRNN) for audio event detection. Our goal is to improve audio event detection performance and recognize target audio 
events that have different lengths and accompany the complex audio background. We exploit multi-groups of parallel and serial convolutional
kernels to learn high-level shift invariant features from the time and frequency domains of acoustic samples. A two-layer bi-direction 
gated recurrent unit) based on the recurrent neural network is used  to capture the temporal context from the extracted high-level features.
The proposed method is evaluated on the DCASE2017 challenge dataset. Compared to other methods, the MTF-CRNN achieves one of the best test 
performances for a single model without pre-training and without using a multi-model ensemble approach.
