from pcdet.datasets.dataset import DatasetTemplate

class NuScenesOpenPCDetAdapter(DatasetTemplate):
    def __init__(
            self,
            nuScenes,
            dataset_cfg,
            class_names,
            root_path=".",
            logger=None,
        ):
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                training=False,
                root_path=Path(root_path),
                logger=logger,
            )
            self.nuScenes = nuScenes
    
    def __len__(self):
        return len(self.my_dataset)

    def __getitem__(self, index):
        sample = self.nuScenes[index]
        points = sample["points"].astype(np.float32)
        sample_id = sample["sample_id"]

        input_dict = {
                "points": points,
                "frame_id": sample_id
                }
        return self.prepare_data(data_dict=input_dict)
