import os

from data.image_operations import Slide, read_csv

class CPTAC_Slides:
    def __init__(self, root, slideroot, slide_list):
        self.data_path = root
        self.svs_path = slideroot
        self.data_list = read_csv(os.path.join(self.data_path, slide_list))

        self.slide_obj_dict = {}
        self._make_dataset()

        self.data_list = list(self.data_list['Slide_Name'])

    def _make_dataset(self):
        for idx, record in self.data_list.iterrows():
            slide_data = Slide(self.data_path, self.svs_path, record)

            self.slide_obj_dict[slide_data.slide_name] = slide_data

    def __getitem__(self, index):
        """Returns a dictionary with all attributes for a slide indexed using the argument.
        The attributes of a slide include:
            - Slide name
            - Slide size
            - Slide path
            - Tiles path
            - Patch list
            - Patch Count
        """

        slide_name = self.data_list[index]
        slide_data = self.slide_obj_dict[slide_name]

        return slide_data

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.data_list)
