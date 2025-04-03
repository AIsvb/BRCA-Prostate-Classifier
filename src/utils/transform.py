from torchvision import transforms

def get_eval_transforms(mean, std, target_img_size = -1, pad=4096):
	trsforms = []
	
	trsforms.append(PadToTargetSize(pad))
	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms


class PadToTargetSize:
	def __init__(self, target_size, padding_value=(228,220,230)):

		self.target_size = target_size
		self.padding_value = padding_value
		
	def __call__(self, img):

		width, height = img.size

		x = self.target_size - width
		y = self.target_size - height

		if x == 0 and y == 0:
			return img
		
		padded_img = transforms.Pad((x, y, 0, 0), fill=self.padding_value)(img)
		return padded_img