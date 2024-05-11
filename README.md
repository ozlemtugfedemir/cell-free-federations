Energy Reduction Cell-Free Massive MIMO through Fine-Grained Resource Management
==================

This code package contains a simulation environment, based on Python, that reproduces some of the numerical results in the article

Özlem Tuğfe Demir, Lianet Méndez-Monsanto, Nicola Bastianello, Emma Fitzgerald, and Gilles Callebaut, "Energy Reduction Cell-Free Massive MIMO through Fine-Grained Resource Management", EUCNC 2024.

## Abstract of Article

The physical layer foundations of cell-free massive MIMO (CF-mMIMO) have been well-established. As a next step, researchers are investigating practical and energy-efficient network implementations. This paper focuses on multiple sets of access points (APs) where user equipments (UEs) are served in each set, termed a federation, without inter-federation interference. The combination of federations and CF-mMIMO shows promise for highly-loaded scenarios. Our aim is to minimize the total energy consumption while adhering to UE downlink data rate constraints. The energy expenditure of the full system is modelled using a detailed hardware model of the APs. We jointly design the AP-UE association variables, determine active APs, and assign APs and UEs to federations. To solve this highly combinatorial problem, we develop a novel alternating optimization algorithm. Simulation results for an indoor factory demonstrate the advantages of considering multiple federations, particularly when facing large data rate requirements. Furthermore, we show that adopting a more distributed CF-mMIMO architecture is necessary to meet the data rate requirements. Conversely, if feasible, using a less distributed system with more antennas at each AP is more advantageous from an energy savings perspective.
