# Motivation
Scenario generation is fundamental to planning, designing, and optimizing decision-making processes in modern power systems with high penetration of renewable energy sources.In previous studies, weather was often categorized based on human judgment, which is overly simplistic and overlooks the intrinsic relationship between weather information at different times and power generation. In our proposed cDDPM model, we utilize weather conditions extracted through an attention mechanism as conditional information for the DDPM, thereby constructing a continuous conditional input space that allows us to focus on more realistic weather information. This model can not only generate scenarios based on historical conditional information but also make precise inferences about unknown weather conditions.
# Run experiments
First, run train.py to train the model, then run generation.py to generate samples, and finally run evaluate.py to compute various metrics to assess the model's performance.
# Language and Framework
All the code is written in Python and runs in PyTorch.
# Dataset
All datasets in these paper are produced and processed from https://dkasolarcentre.com.au/download?location=alice-springs.
# Code References
Thank you for the following codes：
1.https://github.com/chennnnnyize/Renewables_Scenario_Gen_GAN
2.https://github.com/jonathandumas/generative-models
3.https://github.com/EstebanHernandezCapel/DDPM-Power-systems-forecasting
