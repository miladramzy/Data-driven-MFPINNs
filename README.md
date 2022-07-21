# Data-driven-MFPINNs

This repository contains the code and data accompanying the paper entitled "[A Data-driven Multi-fidelity Physics-informed Learning Framework for Smart Manufacturing: A Composites Processing Case Study]"(https://ieeexplore.ieee.org/document/9816983) authored by Milad Ramezankhani; Amir Nazemi; Apurva Narayan; Heinz Voggenreiter; Mehrtash Harandi; Rudolf Seethaler and Abbas S. Milani.

## Abstract
Despite the successful implementations of physics-informed neural networks in different scientific domains, it has been shown that for complex nonlinear systems, achieving an accurate model requires extensive hyperparameter tuning, network architecture design, and costly and exhaustive training processes. To avoid such obstacles and make the training of physics-informed models less precarious, in this paper, a data-driven multi-fidelity physics-informed framework is proposed based on transfer learning principles. The framework incorporates the knowledge from low-fidelity (auxiliary) systems and limited labeled data from target (actual) system to significantly improve the performance of conventional physics-informed models. While minimizing the efforts of designing a complex task-specific network for the problem at hand, the proposed settings guide the physics-informed model towards a fast and efficient convergence to a global optimum. An adaptive weighting method is utilized to further enhance the optimization of the model's composite loss function during the training process. A data-driven strategy is also introduced for maintaining high performance in subdomains with significant divergence between low- and high-fidelity behaviours. The heat transfer of composite materials undergoing a cure cycle is investigated as a case study to demonstrate the proposed framework's superior performance compared to conventional physics-informed models.

## Citation
  @INPROCEEDINGS{9816983,
  author={Ramezankhani, Milad and Nazemi, Amir and Narayan, Apurva and Voggenreiter, Heinz and Harandi, Mehrtash and Seethaler, Rudolf and Milani, Abbas S.},
  booktitle={2022 IEEE 5th International Conference on Industrial Cyber-Physical Systems (ICPS)}, 
  title={A Data-driven Multi-fidelity Physics-informed Learning Framework for Smart Manufacturing: A Composites Processing Case Study}, 
  year={2022},
  volume={},
  number={},
  pages={01-07},
  doi={10.1109/ICPS51978.2022.9816983}}
