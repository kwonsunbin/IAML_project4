import os
import pretty_midi
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_midi_paths(data_path):
    """
    :param data_path: data directory
    :return: list of piano rolls
    """
    paths = []
    for path, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.midi'):
                paths.append(os.path.join(path, file))
    return paths


def midi_to_pianoroll(path):
    """
    :param path: midi path
    :return: piano roll of shape [256, 128] - [frames(sec*fs), notes]
    """
    pm = pretty_midi.PrettyMIDI(path)
    pianoroll = pm.get_piano_roll(fs=8)
    pianoroll = np.array(pianoroll, dtype=np.int32).T
    if pianoroll.shape[0] > 256:
        raise NotImplementedError
    elif pianoroll.shape[0] < 256:
        mask = 256 - pianoroll.shape[0]
        return np.pad(pianoroll, [(0, mask), (0, 0)], 'constant')
    else:
        return pianoroll


def save_pianoroll_to_midi(piano_roll, filename, fs=8, program=0):
    """
    Convert a Piano Roll array into a midi file with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(frames, 128), dtype=int
        Piano roll of one instrument
    filename : save path
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    """
    piano_roll = piano_roll.T
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    pm.write(filename)


class PianoDataset(Dataset):
    def __init__(self, audio_dir='audio'):
        super(PianoDataset, self).__init__()

        self.audio_dir = audio_dir
        self.midi_paths = get_midi_paths(self.audio_dir)

    def __len__(self):
        return len(self.midi_paths)

    def __getitem__(self, idx):
        pianoroll = midi_to_pianoroll(self.midi_paths[idx])
        return pianoroll


if __name__ == '__main__':
    dataset = PianoDataset(audio_dir='audio')
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    first_batch = data_loader.__iter__().__next__()
    pm = first_batch
    print(pm.shape)
    save_pianoroll_to_midi(pm[0], "example.midi", fs=8, program=0)
