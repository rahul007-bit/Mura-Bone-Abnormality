## Installation Steps
1. Clone the repository
2. Install the requirements
3. Run the app.py file
4. Open localhost:5000 in your browser
5. Enjoy the application

### Clone the repository
```bash
git clone https://github.com/rahul007-bit/Mura-Bone-Abnormality.git
```

### Install the requirements
- create a virtual environment
```bash
python3 -m venv env
```
- activate the virtual environment
    - for windows
    ```bash
    env\Scripts\activate
    ```
    - for linux and mac
    ```bash
    source env/bin/activate
    ```

- install the requirements
```bash
pip install -r requirements.txt
```

### Download the Required model
```bash
curl -LO https://huggingface.co/KhalfounMehdi/vit_musculoskeletal_abnormality_detection_mura_224px_16bs_20ep/resolve/main/pytorch_model.bin\?download\=true --progress-bar
```

### Download the required dataset

- download MURA Dataset
    - for windows
        ```bash
        wget https://cs.stanford.edu/group/mlgroup/MURA-v1.1.zip
        ```
    - for linux and mac
        ```bash
        # Get the dataset
        wget -cq https://cs.stanford.edu/group/mlgroup/MURA-v1.1.zip
        ```

- unzip the dataset

    ```bash
    # Unzip the dataset
    !unzip -qq MURA-v1.1.zip
    ```

### Run the app.py file
```bash
streamlit run app.py
```

### Open localhost:5000 in your browser
- go to your browser
- type localhost:5000 in the address bar
- hit enter 😂
- and ta da 🎉🎉🎉🎉🎉🎉🎉🎉🎉

