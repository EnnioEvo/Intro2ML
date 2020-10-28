1) Run generate_validation.py in order to generate val_triplets.txt
	Note: this step can be avoided setting submit=True in main.py and commenting the lines depending on val_triplets.txt
2) Extract food images in data/food/
3) Run generate_features.py
	Note: this step takes significantly less using a GPU
4) Run main.py