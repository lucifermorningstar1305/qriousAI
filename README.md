# Can small mobile models understand text?

In this project we tried to observe if mobile models have the capacity of understanding text or not. Thereby we trained we a **MobileNet** accompanied with **Lite-Transformers** following the **CLIP** and **CLIP-Lite** methods on $10\%$ of the MS-COCO Captions dataset. And the result of the training is as follows:

### Zero-shot Capabilities

![zero_shot](https://github.com/lucifermorningstar1305/qriousAI/blob/main/media/zero_shot.png)

The model shows the capability of possessing Zero-Shot classification even though having only $44M$ parameters compared to the **CLIP** model.

### Visual-Text grounding
![visual_ground](https://github.com/lucifermorningstar1305/qriousAI/blob/main/media/visual_ground.png)

When the model was prompted with recognizing specific object in an image, the model was able to highlight that specific object, as shown in the above figure, indicating that the model possess the capability of visual-text grounding. 

## Run this project

1. Clone the repo by pasting the following text into your terminal:
	`git clone https://github.com/lucifermorningstar1305/qriousAI.git`

2. Setup the training environment using the following command:
	`conda env create -f qrious_env.yml`

3. To train the models run the following command:

	```bash
	python train_coco.py\ 
	--train_data_path <path where you have stored the mscoco captions training dataset as a csv>\ 
	--val_data_path <path where you have stored the mscoco captions validation dataset as a csv>\ 
	--config_path ./configs/config.yaml\ 
	--checkpoint_filename <filename for your checkpoint>\ 
	--max_epochs 500\ --early_stopping_patience 5\ 
	--data_size .1\ 
	--accumulate_grad_batches 10```

4. To evaluate the models run the following command:
```bash
python evaluate_models.py --root_dir <path to store the evaluation dataset>\ 
--dataset <name of pytorch dataset to download>\ 
--model_checkpoint <checkpoint of model to evaluate>\
--config_path ./configs/config.yaml\
--prompt "A photo of a"
```

## Results on Standard Benchmark
![results](https://github.com/lucifermorningstar1305/qriousAI/blob/main/media/zero_shot_accs_chart.png)
The above chart indicates the `top-1` and `top-5` accuracy of the model on standard Computer vision benchmarks. The reason for such low scores compared to the original **CLIP** is the less amount of data being used for training. With more data these accuracies can be enhanced.

## Inaccurate Results

There are cases where the model does fail to perform zero-shot or locate specific objects in an image. The following figure highlights those cases:

![inacc_results](https://github.com/lucifermorningstar1305/qriousAI/blob/main/media/inaccurate_results.png)

The first figure showcases the failing of zero-shot classification, where given an image of a *modern concept car*, the model classifies the image as a *jet*. 

The second figure showcases the failing of visual-text grounding of the model, given the prompt "The person in the image", the model highlights the *husky* in the image.

This inaccuracies indicate the model needs to trained with more data in order for it to understand objects with better accuracies.


