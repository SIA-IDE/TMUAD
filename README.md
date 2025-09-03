# TMUAD: Enhancing Logical Capabilities in Unified Anomaly Detection Models with a Text Memory Bank
## Overview

<table>
  <tr>
    <td><img src="./demo/main_framework_diagram.jpg" width="100%"></td>
  </tr>
</table>

To unify logical and structural anomaly detection, we propose three complementary strategies: 
1. Retain patch-level image features to construct a patch-level image memory bank $M_P$, thereby preserving the model’s capability to detect structural anomalies. 
2. Introduce a logic-aware textual description of images to significantly enhance logical anomalies detection of unified anomaly detection frameworks by constructing a class-level text memory bank $M_T$. 
3. Introduce object-level image features from segmented images to build an object-level image memory bank $M_O$, which compensates for the inability of patch-level features to fully capture the shape and structure of individual objects.