from main import build_events_data_dict
from utils import video_utils
from utils.main_utils import get_parameters, RunParameters

if __name__ == '__main__':
    parameters: RunParameters
    input_folder, _, _, _, data_path, parameters = get_parameters(is_vid_names=True)
    videos_dict = build_events_data_dict(data_path, parameters.vid_names, parameters.event_number)

    result = {}
    for fish_name in videos_dict.keys():
        max_n_frames = 0
        for vid_data_dict in videos_dict[fish_name]['videos']:
            vidname = vid_data_dict['name']

            video, _, _, n_frames, _ = video_utils.open(input_folder, vidname)
            video_utils.release(video, visualize_movie=False)

            if max_n_frames < n_frames:
                max_n_frames = n_frames

        result[fish_name] = max_n_frames

    if len(result.keys()) == 1:
        print(result[list(result.keys())[0]])
    else:
        print(result)


