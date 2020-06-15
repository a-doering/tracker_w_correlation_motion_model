from .mot_sequence import MOT17Sequence
from ..config import get_output_dir

#TODO more imports

class MOTcorrelation(MOT17Sequence):
    """Multiple object tracking dataset.
    
    This class builds samples for training a siamese net. 
    #TODO add more
    """

    def __init__(self, seq_name, split, vis_treshold)

        super().__init__(seq_name, vis_treshold=vis_treshold)

        #TODO more inits

		self.build_samples()

		if split == 'train':
			pass
		elif split == 'small_train':
			self.data = self.data[0::5] + self.data[1::5] + self.data[2::5] + self.data[3::5]
		elif split == 'small_val':
			self.data = self.data[4::5]
		else:
			raise NotImplementedError("Split: {}".format(split))  
