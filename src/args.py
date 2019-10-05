import argparse


def add_preprocess_args(parser):

    prep_parser = parser.add_argument_group("Preprocess options")

    prep_parser.add_argument('--scaling', dest='scaling', action='store_true', help='Feature scaling or not')
    prep_parser.add_argument('--augment', dest='augment', action='store_true',
                        help='Use random tempo and gain perturbations.')
    prep_parser.add_argument('--duration', default=1.0, type=float, help='Duration of one EEG dataset')
    prep_parser.add_argument('--window-size', default=4.0, type=float, help='Window size for spectrogram in seconds')
    prep_parser.add_argument('--window-stride', default=2.0, type=float, help='Window stride for spectrogram in seconds')
    prep_parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
    prep_parser.add_argument('--spect', dest='spect', action='store_true', help='Use spectrogram as input')
    prep_parser.add_argument('--sample-rate', default=400, type=int, help='Sample rate')
    prep_parser.add_argument('--num-eigenvalue', default=0, type=int,
                             help='Number of eigen values to use from spectrogram')
    prep_parser.add_argument('--l-cutoff', default=0.01, type=float, help='Low pass filter')
    prep_parser.add_argument('--h-cutoff', default=10000.0, type=float, help='High pass filter')
    prep_parser.add_argument('--mfcc', dest='mfcc', action='store_true', help='MFCC')
    prep_parser.add_argument('--to_1d', dest='to_1d', action='store_true', help='Preprocess inputs to 1 dimension')

    return parser
