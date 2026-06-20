#!/bin/bash
while ! grep -q "Extraction complete" ~/.gemini/antigravity-ide/brain/13facef4-8318-4dd3-9dde-e67fedd6b16d/.system_generated/tasks/task-1065.log; do
    sleep 5
done
echo "Radiomics complete! Running Model 3 notebook..."
jupyter nbconvert --to notebook --execute --inplace notebooks/03_Model3_Imaging_TCGA.ipynb
echo "Model 3 complete! Running Final Fusion notebook..."
jupyter nbconvert --to notebook --execute --inplace notebooks/04_Late_Fusion_3Modality.ipynb
echo "All notebooks successfully executed!"
