"""
Configuration of the experiments.
"""

config = {
	'pgd': {
	'epsilon': 8/255.,
	'alpha': 2/255.,
	'step': 40
	},
	'occlusion': {
	'epsilon': 255/255.,
	'alpha': 4/255.,
	'step': 200
	},
	'facemask': {
	'epsilon': 255/255.,
	'alpha': 4/255.,
	'step': 200,
	'width': 16, 
	'height': 8
	},
	'threshold': {
	'vggface': 0.6408,
	'facenet': 0.4338
	},
	'batch_size': 50
}

config_colab = {
	'pgd': {
	'epsilon': 8/255.,
	'alpha': 2/255.,
	'step': 40
	},
	'occlusion': {
	'epsilon': 255/255.,
	'alpha': 4/255.,
	'step': 200
	},
	'facemask': {
	'epsilon': 255/255.,
	'alpha': 4/255.,
	'step': 200,
	'width': 16, # width of color grids on face masks
	'height': 8 # heights of color grids on face masks
	},
	'threshold': {
	'vggface': 0.6408,
	'facenet': 0.4338
	},
	'batch_size': 50
	,
	'VGG2TEST' :{
		'train_ds' :'VGG2',
		'test_ds':'VGG2TEST'
	},
	'drive': '/content/drive/MyDrive/GOOGLE_COLAB'
	,

	'VGG-FACE':{
		'convolve': 'pretrained_vggface.t7',
		'fc': 'vggface_closed.pt'
	}
	,
	'lfw':{
		'name_src': 'lfw',
		'transform_size': [(112, 112), (160, 160), (112, 96)]
	}
	,'frame_atack':{
		'masks': ['eyeglass', 'facemask', 'sticker']
	}
}