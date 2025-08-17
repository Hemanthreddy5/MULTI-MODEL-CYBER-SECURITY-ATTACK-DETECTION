import os
from utils.dataset_generator import generate_synthetic_cybersecurity_data, save_dataset

def main():
    os.makedirs('data', exist_ok=True)
    
    print("Generating synthetic cybersecurity dataset...")
    df = generate_synthetic_cybersecurity_data(
        n_samples=10000,  
        attack_ratio=0.36  
    )
    
    save_dataset(df)

if __name__ == "__main__":
    main() 