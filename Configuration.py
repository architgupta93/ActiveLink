"""
Create configuration structure for different environments with the following fields
    - N_TETRODES: Number of tetrodes which are interesting
    - TETRODE_LIST: List of the tetrode list indices
    - CLUSTER_FILENAME: Name/Location of the cluster file for the data we
      are using. Does not have to be supplied as the program lets you
      choose it using a GUI.

    [OPTIONAL ARGUMENTS]
    TODO: These haven't been added yet.
    - SPEED_THRESHOLD
"""
import os
import logging
import configparser
import QtHelperUtils
import xml.etree.ElementTree as ET

DEFAULT_CONFIG_FILE='config/default.ini'
MODULE_IDENTIFIER="[Configuration] "
MANUAL_STIM_DURATION = 10
MANUAL_STIM_INTER_PULSE_INTERVAL = 1.0

EXPERIMENT_DAY_20190307__INTERESTING_CLUSTERS_A = [1, 2, 25, 30, 32, 59]
EXPERIMENT_DAY_20190307__INTERESTING_CLUSTERS_B = [16, 34, 39, 43, 56]
EXPERIMENT_DAY_20190307__ALL_INTERESTING_CLUSTERS = [1, 2, 16, 25, 30, 32, 34, 39, 43, 56, 59]

def read_cluster_file(filename=None, tetrodes=None):
    """
    Reads a cluster file and generates a list of tetrodes that have cells and
    all the clusters on that tetrode.

    :filename: XML file containing clustering information.
    :tetrodes: Which tetrodes to look at in the cluster file.
    :returns: A dictionary giving valid cluster indices for each tetrode.
    """
    if filename is None:
        filename = QtHelperUtils.get_open_file_name(file_format="Cluster File (*.trodesClusters)", \
                message="Select Cluster file")
        if not filename:
            return None

    try:
        cluster_tree = ET.parse(filename)
    except (FileNotFoundError, IOError) as err:
        print(err)
        return (0, {})

    if __debug__:
        print(MODULE_IDENTIFIER + 'Read cluster file ' + filename)

    # The file is organized as:
    # [ROOT] SpikeSortInfo
    #       -> PolygonClusters
    #           -> ntrode (nTrodeID)
    #               -> cluster (clusterIndex)

    n_trode_to_cluster_idx_map = {}
    raw_cluster_idx = 0
    # Some unnecessary accesses to get to tetrodes and clusters
    tree_root = cluster_tree.getroot()
    polygon_clusters = tree_root[0]
    ntrode_list = list(polygon_clusters)
    if tetrodes is None:
        tetrodes = range(1, 1+len(ntrode_list))

    if __debug__:
        print(MODULE_IDENTIFIER + '%d tetrode(s) in cluster file.'%len(tetrodes))
        print(tetrodes)

    for t_i in tetrodes:
        # Offset by 1 because Trodes tetrodes start with 1!
        ntrode = ntrode_list[t_i-1]
        n_clusters_on_ntrode = 0
        tetrode_idx = 1 + int(ntrode.get('nTrodeIndex'))
        tetrode_num = ntrode.get('nTrodeID')
        if len(list(ntrode)) == 0:
            # Has no clusters on it
            continue

        # TODO: These indices go from 1.. N. Might have to switch to 0.. N if
        # that is what spike data returns.
        cluster_idx_to_id_map = {}
        for cluster in ntrode:
            local_cluster_idx = cluster.get('clusterIndex')
            cluster_idx_to_id_map[int(local_cluster_idx)] = raw_cluster_idx
            raw_cluster_idx += 1
            n_clusters_on_ntrode += 1
        n_trode_to_cluster_idx_map[tetrode_idx] = cluster_idx_to_id_map
        if n_clusters_on_ntrode == 0:
            n_trode_to_cluster_idx_map.pop(tetrode_idx, None)

    # Final value of raw_cluster_idx is a proxy for the total number of units we have
    logging.info(MODULE_IDENTIFIER + "Cluster map...\n%s"% n_trode_to_cluster_idx_map)
    return raw_cluster_idx, n_trode_to_cluster_idx_map

def get_open_field_configuration(filename=None):
    """
    Return configuration for the open field.
    """
    if filename is None:
        filename = QtHelperUtils.get_open_file_name(file_format="Cluster File (*.trodesClusteres)", \
                message="Select Cluster file")

    configuration = configparser.ConfigParser()
    try:
        configuration.read(filename)
    except (FileNotFoundError, IOError) as err:
        print('Unable to read configuration file %s. Using defaults.'%filename)
        print(err)
        configuration.read(DEFAULT_CONFIG_FILE)

class Config(object):

    """
    Get the configuration for running ripple/replay interruption
    """

    def __init__(self):
        """TODO: to be defined1. """
