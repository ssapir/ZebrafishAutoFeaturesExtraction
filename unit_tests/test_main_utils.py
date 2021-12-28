import unittest

from utils.main_utils import get_info_from_event_name


class InfoFromEventName(unittest.TestCase):
    def test_full_fish_name(self):
        no_event_id_names = ['20200723-f1.raw', '20200720-f2-whole-movie.raw', '20210106-testfish.raw']
        event_id_1_names = ['20200720-f3-1.raw', '20210106-testfish-1.raw']

        expected_name = ['20200723-f1', '20200720-f2', '20210106-testfish']
        for i in range(len(no_event_id_names)):
            name = no_event_id_names[i]
            fish_name, event_number = get_info_from_event_name(name)
            self.assertEqual(event_number, -1)
            self.assertEqual(fish_name, expected_name[i])
        expected_name = ['20200720-f3', '20210106-testfish']
        for i in range(len(event_id_1_names)):
            name = event_id_1_names[i]
            fish_name, event_number = get_info_from_event_name(name)
            self.assertEqual(event_number, 1)
            self.assertEqual(fish_name, expected_name[i])

    def test_event_fish_name(self):
        event_names = ['20200720-f3-1.raw', '20200720-f3-45.raw', '20210106-testfish-2.raw']
        expected_name = ['20200720-f3', '20200720-f3', '20210106-testfish']
        expected_events = [1,  45, 2]
        for i in range(len(event_names)):
            name = event_names[i]
            fish_name, event_number = get_info_from_event_name(name)
            self.assertEqual(event_number, expected_events[i])
            self.assertEqual(fish_name, expected_name[i])


if __name__ == '__main__':
    unittest.main()
