# Data-driven-MFPINNs

This repository contains the code and data accompanying the paper entitled "[A Data-driven Multi-fidelity Physics-informed Learning Framework for Smart Manufacturing: A Composites Processing Case Study]"(https://ieeexplore.ieee.org/document/9816983) authored by Milad Ramezankhani; Amir Nazemi; Apurva Narayan; Heinz Voggenreiter; Mehrtash Harandi; Rudolf Seethaler and Abbas S. Milani.

## Abstract
Despite the successful implementations of physics-informed neural networks in different scientific domains, it has been shown that for complex nonlinear systems, achieving an accurate model requires extensive hyperparameter tuning, network architecture design, and costly and exhaustive training processes. To avoid such obstacles and make the training of physics-informed models less precarious, in this paper, a data-driven multi-fidelity physics-informed framework is proposed based on transfer learning principles. The framework incorporates the knowledge from low-fidelity (auxiliary) systems and limited labeled data from target (actual) system to significantly improve the performance of conventional physics-informed models. While minimizing the efforts of designing a complex task-specific network for the problem at hand, the proposed settings guide the physics-informed model towards a fast and efficient convergence to a global optimum. An adaptive weighting method is utilized to further enhance the optimization of the model's composite loss function during the training process. A data-driven strategy is also introduced for maintaining high performance in subdomains with significant divergence between low- and high-fidelity behaviours. The heat transfer of composite materials undergoing a cure cycle is investigated as a case study to demonstrate the proposed framework's superior performance compared to conventional physics-informed models.

## Citation
M. Ramezankhani et al., "A Data-driven Multi-fidelity Physics-informed Learning Framework for Smart Manufacturing: A Composites Processing Case Study," 2022 IEEE 5th International Conference on Industrial Cyber-Physical Systems (ICPS), 2022, pp. 01-07, doi: 10.1109/ICPS51978.2022.9816983.

## Results
![MPINNS_ICPS](https://user-images.githubusercontent.com/35118419/180121216-ae86431e-3423-4a18-9c65-70177e4e0e88.png)
Fig. 1. Schematic of the proposed MFPINN framework. First, PINN_L is trained using the low-fidelity data (x_L,y_L) and PDEs (L_L). Next, to train PINN_H, the learned low-fidelity knowledge (PINN_L’s predictions) along with the high-fidelity data (x_H,y_H) and PDEs (L_H) are utilized. Automatic differentiation is used to calculate the derivates of the network predictions (yellow circles) needed for calculating the PDE loss terms.

![image](https://user-images.githubusercontent.com/35118419/180121392-9bc9266e-96e6-4ed7-a171-cffbf0c106ed.png)
Fig. 2. One-hold cure cycle specifications used for the case study. The red and blue curves show the temperature development in the middle section of Composite 1 (low-fidelity model) and Composite 2 (high-fidelity test).![image](https://user-images.githubusercontent.com/35118419/180121415-297d5a38-1aae-4f30-8960-10ca4156b1fe.png)

![image](https://user-images.githubusercontent.com/35118419/180121454-41a11f57-6771-484a-aeaf-34398e59d66e.png)
![image](https://user-images.githubusercontent.com/35118419/180121464-73281672-cffe-4070-8112-3be339fa32ca.png)
Fig. 3. Effect of labeled high-fidelity data in the PINN model performance. Absolute error field (°C) between the true temperature and the prediction of the trained PINN model a) with no labeled data (relative L_2=0.13), and b) with 100 FE labeled data are shown (relative L_2=0.051). For a fair visual comparison, the same color bar scale is used for both figures.

![image](https://user-images.githubusercontent.com/35118419/180121528-4577636d-1485-4bf3-9f5b-06059caec453.png)
![image](https://user-images.githubusercontent.com/35118419/180121534-5eaeec89-2cfd-461c-935c-e7650b540ffd.png)
![image](https://user-images.githubusercontent.com/35118419/180121536-527b8b3f-c55d-4b73-9436-c8f1439cd044.png)
![image](https://user-images.githubusercontent.com/35118419/180121547-c5f46e9c-0e12-40ed-937c-6887b005ae32.png)
Fig. 4. Performance of MFPINN model in composites’ heat transfer case study. Absolute error between the true (FE) temperature and model’s prediction is illustrated for a) vanilla PINN, b) MFPINN with no labeled high-fidelity data and d) MFPINN with labeled high-fidelity data. c) shows the experimental design described in section III.C.

![image](https://user-images.githubusercontent.com/35118419/180121581-254917f7-3eab-43ef-afab-7c4e626fd4cc.png)
Fig. 5. Prediction of PINN and MFPINN models for part’s temperature at the center (Composite 2). FE result is also presented for comparison. The zoom-in view emphasizes the MFPINN model improvement as a result of adding labeled high-fidelity data.





