# Data-driven-MFPINNs

This repository contains the code and data accompanying the paper entitled "[A Data-driven Multi-fidelity Physics-informed Learning Framework for Smart Manufacturing: A Composites Processing Case Study]"(https://ieeexplore.ieee.org/document/9816983) authored by Milad Ramezankhani; Amir Nazemi; Apurva Narayan; Heinz Voggenreiter; Mehrtash Harandi; Rudolf Seethaler and Abbas S. Milani.

## Abstract
Despite the successful implementations of physics-informed neural networks in different scientific domains, it has been shown that for complex nonlinear systems, achieving an accurate model requires extensive hyperparameter tuning, network architecture design, and costly and exhaustive training processes. To avoid such obstacles and make the training of physics-informed models less precarious, in this paper, a data-driven multi-fidelity physics-informed framework is proposed based on transfer learning principles. The framework incorporates the knowledge from low-fidelity (auxiliary) systems and limited labeled data from target (actual) system to significantly improve the performance of conventional physics-informed models. While minimizing the efforts of designing a complex task-specific network for the problem at hand, the proposed settings guide the physics-informed model towards a fast and efficient convergence to a global optimum. An adaptive weighting method is utilized to further enhance the optimization of the model's composite loss function during the training process. A data-driven strategy is also introduced for maintaining high performance in subdomains with significant divergence between low- and high-fidelity behaviours. The heat transfer of composite materials undergoing a cure cycle is investigated as a case study to demonstrate the proposed framework's superior performance compared to conventional physics-informed models.

## Citation
M. Ramezankhani et al., "A Data-driven Multi-fidelity Physics-informed Learning Framework for Smart Manufacturing: A Composites Processing Case Study," 2022 IEEE 5th International Conference on Industrial Cyber-Physical Systems (ICPS), 2022, pp. 01-07, doi: 10.1109/ICPS51978.2022.9816983.

## Results
![MPINNS_ICPS](https://user-images.githubusercontent.com/35118419/180121216-ae86431e-3423-4a18-9c65-70177e4e0e88.png | width=100px)
<img src="[https://github.com/favicon.ico](https://user-images.githubusercontent.com/35118419/180121216-ae86431e-3423-4a18-9c65-70177e4e0e88.png)" width="150" height="280">
Fig. 1. Schematic of the proposed MFPINN framework. First, PINN_L is trained using the low-fidelity data (x_L,y_L) and PDEs (L_L). Next, to train PINN_H, the learned low-fidelity knowledge (PINN_L’s predictions) along with the high-fidelity data (x_H,y_H) and PDEs (L_H) are utilized. Automatic differentiation is used to calculate the derivates of the network predictions (yellow circles) needed for calculating the PDE loss terms.


![CC](https://user-images.githubusercontent.com/35118419/180121750-4d415ee9-3190-4ff9-9eec-da20f5d18c29.png)

Fig. 2. One-hold cure cycle specifications used for the case study. The red and blue curves show the temperature development in the middle section of Composite 1 (low-fidelity model) and Composite 2 (high-fidelity test).![image](https://user-images.githubusercontent.com/35118419/180121415-297d5a38-1aae-4f30-8960-10ca4156b1fe.png)

![Residual_PDEonly_702_2cm_v2](https://user-images.githubusercontent.com/35118419/180121892-687e1ed9-b8e4-4a86-b807-0309d03e48b2.png)
![Error_PINNwithData_702_2cm_v1](https://user-images.githubusercontent.com/35118419/180122369-6bbf9901-47b9-434d-a894-3c10da17e825.png)

Fig. 3. Effect of labeled high-fidelity data in the PINN model performance. Absolute error field (°C) between the true temperature and the prediction of the trained PINN model a) with no labeled data (relative L_2=0.13), and b) with 100 FE labeled data are shown (relative L_2=0.051). For a fair visual comparison, the same color bar scale is used for both figures.

![Residual_PDEonly_702_2cm_v2](https://user-images.githubusercontent.com/35118419/180121892-687e1ed9-b8e4-4a86-b807-0309d03e48b2.png)
![Residual_MFPINN-LFOnly_702_2cm_v2](https://user-images.githubusercontent.com/35118419/180122654-7e9312e0-68f7-4a19-b187-1c9c184ae366.png)
![Temp_dist_MFPINN-PDE_LF_Data_702_2cm](https://user-images.githubusercontent.com/35118419/180122036-7f20cb81-0dea-4c2c-8254-81615a879bf9.png)
![Error_MFPINN-PDE_LF_Data_702_2cm](https://user-images.githubusercontent.com/35118419/180121986-a4901294-f1d7-4d6f-8839-f7125d59c842.png)

Fig. 4. Performance of MFPINN model in composites’ heat transfer case study. Absolute error between the true (FE) temperature and model’s prediction is illustrated for a) vanilla PINN, b) MFPINN with no labeled high-fidelity data and d) MFPINN with labeled high-fidelity data. c) shows the experimental design described in section III.C.

![Summary_Middle](https://user-images.githubusercontent.com/35118419/180122171-ca1f3869-b343-4066-872c-6d2235641019.png)

Fig. 5. Prediction of PINN and MFPINN models for part’s temperature at the center (Composite 2). FE result is also presented for comparison. The zoom-in view emphasizes the MFPINN model improvement as a result of adding labeled high-fidelity data.





