clr-all-img:
	rm -r data/images/faces/*
	rm -r data/images/*.jpg
	echo "images cleared"

init:
	mkdir data/images
	mkdir data/images/faces

run:
	streamlit run main.py
