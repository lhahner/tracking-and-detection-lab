@TRACKER.register("AB3DMOT")
class AB3DMOT:
    def __track_sequence_frame_by_class(self, tracking_classes, sequence_name, frame_results, id_start):
        for class_name in tracking_classes:
            tracker = self.__create_tracker(class_name, id_start)
            sequence_file = self.__formatted_category_dir(class_name) / f"{sequence_name}.txt"
            dets, has_detections = load_detection(str(sequence_file)) if sequence_file.exists() else ([], False)

            for frame_number in self.__frame_order_by_sequence[sequence_name]:
                dets_frame = self.__frame_detections(dets,
                                                     has_detections,
                                                     frame_number)
                results, _ = tracker.track(dets_frame, frame_number, sequence_name)
                frame_results[frame_number]["tracks"].extend(
                    self.__results_to_tracks(
                        results[0],
                        class_name,
                        self.__frame_metadata_by_sequence[sequence_name][frame_number],
                    )
                )
            id_start = max(id_start, tracker.ID_count[0])
        return id_start, frame_results