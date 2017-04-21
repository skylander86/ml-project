from argparse import ArgumentParser
from collections import Counter
import json
import logging

from tabulate import tabulate

from .uriutils import URIFileType
from .timer import Timer

__all__ = ['load_instances']

logger = logging.getLogger(__name__)


def load_instances(instance_files, labels_field='labels', limit=None):
    total_count = 0
    labels_freq = Counter()
    for f in instance_files:
        freq = Counter()
        count = 0
        timer = Timer()
        for lineno, line in enumerate(f, start=1):
            o = json.loads(line)

            labels = o.get(labels_field)
            if labels is not None:
                del o[labels_field]
                if labels: freq.update(labels)
                else: freq['<none>'] += 1
            #end if

            yield (o, labels)
            count += 1
            if count == limit: break
        #end for
        logger.info('{} instances read from file <{}> {}.'.format(count, f.name, timer))

        labels_freq += freq
        total_count += count
        logger.info('Label frequencies for <{}>:\n{}'.format(f.name, tabulate(freq.most_common() + [('Labels total', sum(freq.values())), ('Cases total', count)], headers=('Label', 'Freq'), tablefmt='psql')))

        if count == limit: break
    #end for

    logger.info('Total label frequencies:\n{}'.format(tabulate(labels_freq.most_common() + [('Labels total', sum(labels_freq.values())), ('Cases total', count)], headers=('Label', 'Freq'), tablefmt='psql')))
#end def


def main():
    parser = ArgumentParser(description='Quick utility for getting instance information.')
    parser.add_argument('instances', type=URIFileType(encoding='ascii'), nargs='+', metavar='<instances>', help='List of instance files to get information about.')
    A = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    labels_freq = Counter()
    for f in A.instances:
        X, Y_labels = list(zip(*load_instances([f], labels_field='labels')))
        freq = Counter(label for labels in Y_labels for label in labels)
        freq['<none>'] = sum(1 for labels in Y_labels if not labels)

        logger.info('Label frequencies for <{}>:\n{}'.format(f.name, tabulate(freq.most_common() + [('Labels total', sum(freq.values())), ('Cases total', len(X))], headers=('Label', 'Freq'), tablefmt='psql')))
        labels_freq += freq
    #end for

    if len(A.instances) > 1:
        logger.info('Total label frequencies:\n{}'.format(tabulate(labels_freq.most_common() + [('Labels total', sum(labels_freq.values())), ('Cases total', len(X))], headers=('Label', 'Freq'), tablefmt='psql')))
#end def


if __name__ == '__main__': main()
