<p align="center">
  <h1 align="center">ELite: Ephemerality meets LiDAR-based Lifelong Mapping</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=n15gehEAAAAJ"><strong>Hyeonjae Gil*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=t_5U_98AAAAJ"><strong>Dongjae Lee*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=9mKOLX8AAAAJ"><strong>Giseop Kim</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=7yveufgAAAAJ"><strong>Ayoung Kim</strong></a>
  </p>
  <p align="center">(* Equal Contribution)</p>
  <!-- <h3 align="center"><a href="https://arxiv.org/abs/2502.13452">Arxiv</a> | <a href="https://arxiv.org/abs/2502.13452">Paper</a> | <a href="https://www.youtube.com/watch?v=xZwzNgcHqjc">Video</a></h3> -->
  <h3 align="center"><a href="https://arxiv.org/abs/2502.13452">Arxiv</a> | <a href="https://www.youtube.com/watch?v=xZwzNgcHqjc">Video</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <img src="./assets/ELite_main.png" width="80%" />
</p>

## About ELite

**ELite** is a LiDAR-based lifelong mapping framework that integrates *ephemerality*—the likelihood of a point being transient or persistent—into the entire pipeline. Unlike binary classifications of map elements (static vs. dynamic), ELite introduces a two-stage ephemerality concept to represent changes across different time scales:
- **Local ephemerality**: Captures short-term variations (e.g., moving cars vs. parked cars).
- **Global ephemerality**: Represents long-term changes (e.g., parked cars vs. new buildings)

By leveraging ephemerality, ELite seamlessly aligns multiple sessions, removes dynamic objects, and matains accurate maps.

<p align="center">
  <img src="./assets/ELite_pipeline.png" width="80%" />
</p>

## Run
*Code will be available soon!*

## Citation
If you use ELite for any academic work, please cite our paper.
```bibtex
@INPROCEEDINGS { hjgil-2025-icra,
    author={Hyeonjae Gil and Dongjae Lee and Giseop Kim and Ayoung Kim},
    title={Ephemerality meets LiDAR-based Lifelong Mapping},
    booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
    year={2025},
    month={May.},
    address={Atlanta},
}
```

## Contact
If you have any questions, please contact:
- Hyeonjae Gil ([now9728@gmail.com]())
- Dongjae Lee ([dongjae0107@gmail.com]())

## Acknowledgement 
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT)(No. RS-2024-00461409), and in part by the Robotics and AI (RAI) Institute.