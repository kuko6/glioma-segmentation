FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:20220314.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/tensorflow-2.7

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=20.2.4

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'matplotlib~=3.5.0' \
                'pandas~=1.3.0' \
                'scipy~=1.7.0' \
                'numpy~=1.21.0' \
                'azureml-core==1.39.0' \
                'azureml-defaults==1.39.0' \
                'azureml-mlflow==1.39.0.post1' \
                'azureml-telemetry==1.39.0' \
                'tensorflow-gpu~=2.7.0' \
                'onnxruntime-gpu~=1.9.0' \
                'horovod~=0.23.0'
                'keras~=2.7.0' \
                'matplotlib~=3.5.0' \
                'nibabel~=3.2.2' \
                'scikit-learn~=1.0.2' \
                'scikit-image' \
                'opencv-python~=4.5.5.64' \
                'opencv-python-headless~=4.5.5.64' \
                'segmentation-models-3D'

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH