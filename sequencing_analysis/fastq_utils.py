import os, sys
import math
import pickle
import multiprocessing as mp
import tempfile

import numpy as np
import tqdm

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

def analyze_fastq(path_forward, path_reverse = None, path_templates = 'templates.fasta', output_prefix = 'output', fastq_2way = True, max_reads = None, hash_f_start = 35, hash_f_length = 24, hash_r_start = 36, hash_r_length = 24, template_hash_f_start = 23, template_hash_r_start = 24, UMI_f_start = 0, UMI_f_length = 12, UMI_r_start = 0, UMI_r_length = 12, max_hash_defect = 5, min_cluster_abundance = 2e-4, sequencing_accuracy = 0.75, max_reconstruction_depth = 100, reconstruction_method = 'Levenshtein-median', selfalign = True, align_left_marker = None, align_left_position = None, align_right_marker = None, align_right_position = None, mask=None, sample_id = None, full_output = False, verbose = False):
  # Load sequencing data
  if fastq_2way:
    reads_raw = utils.import_fastq_2way(path_forward, path_reverse, max_reads = max_reads, sample_id = sample_id, max_sequence_length = None, verbose = True)
    reads = reads_raw
  else:
    reads_raw = utils.import_fastq(path_forward, max_reads = max_reads, sample_id = sample_id, max_sequence_length = None, verbose = True)
    reads = reads_raw

  # Load template data
  templates = utils.import_fasta(path_templates, sample_id = sample_id, verbose = True)

  # If selfalign is True, convert each 2-way read to a single-sequence read by aligning the forward and reverse reads.
  # Note that this is ignored if fastq_2way was originally set to False, as no self-alignment can occur in that case.
  # The fastq_2way flag is set to False if self-alignment occurs, 
  # to indicate to downstream functions that the reads are no longer 2-way.
  if fastq_2way and selfalign:
    # Align forward and reverse reads with each other
    with mp.Pool(20) as p:
      res_iter = p.imap(utils.fastq_2way_selfalign, reads, chunksize=100)
      reads_selfaligned = [r for r in tqdm.tqdm(res_iter, total=len(reads), desc='Self-aligning reads') if r[1] is not None]
    reads = reads_selfaligned
    fastq_2way = False
  else:
    reads_selfaligned = None
    reads = reads_raw

  # If a left alignment marker (e.g. a master primer sequence) is given, then align on the left 
  # This operation shifts the sequence so the alignment marker appears at the appropriate position of the sequence
  if align_left_marker is not None:
    if fastq_2way:  align_func = utils.fastq_2way_align_to_marker
    else:  align_func = utils.fastq_align_to_marker

    with mp.Pool(20) as p:
      args_lst = [(align_func, {
          'read': r, 
          'marker': align_left_marker, 'position': align_left_position, 
          'side': 'left', 
          'acc_cutoff': sequencing_accuracy, 
          'orient': True
      }) for r in reads]
      res_iter = p.imap(_call_func_starstarargs, args_lst, chunksize=100)
      reads_aligned_left = [r for r in tqdm.tqdm(res_iter, total=len(args_lst), desc='Left-alignment') if r[1] is not None]
    reads = reads_aligned_left
  else:
    reads_aligned_left = None

  # If a right alignment marker is given, then align on the right
  if align_right_marker is not None:
    if fastq_2way:  align_func = utils.fastq_2way_align_to_marker
    else:  align_func = utils.fastq_align_to_marker

    with mp.Pool(20) as p:
      args_lst = [(align_func, {
          'read': r,
          'marker': align_right_marker, 'position': align_right_position,
          'side': 'right',
          'acc_cutoff': sequencing_accuracy,
          'orient': True
      }) for r in reads]
      res_iter = p.imap(_call_func_starstarargs, args_lst, chunksize=100)
      reads_aligned_right = [r for r in tqdm.tqdm(res_iter, total=len(args_lst), desc='Right-alignment') if r[1] is not None]
    reads = reads_aligned_right
  else:
    reads_aligned_right = None

  # Apply the mask at this point, if we have 1-way reads (masks for 2-way reads aren't supported)
  if mask is not None and not fastq_2way:
    reads_masked = [(name, ''.join(nt if mask_bit else 'N' for nt, mask_bit in zip(seq, mask))) for name,seq in reads]
    reads = reads_masked
  else:
    reads_masked = None
 
  # Cluster reads based on their hash sequences
  min_cluster_size = math.ceil(min_cluster_abundance * len(reads))
  cluster_hashes, cluster_reads = cluster_reads_by_hash(
      reads, 
      hash_f_start = hash_f_start, hash_f_length = hash_f_length,
      hash_r_start = hash_r_start, hash_r_length = hash_r_length,
      fastq_2way = fastq_2way,
      max_hash_defect = max_hash_defect,
      max_reconstruction_depth = max_reconstruction_depth,
      reconstruction_method = reconstruction_method,
      min_cluster_size = min_cluster_size,
      sample_id = sample_id, 
      verbose = True,
  )

  # Count cluster sizes by UMI
  cluster_sizes_UMI = calc_cluster_sizes_UMI(
      reads, 
      cluster_reads,
      UMI_f_start = UMI_f_start, UMI_f_length = UMI_f_length,
      UMI_r_start = UMI_r_start, UMI_r_length = UMI_r_length,
      fastq_2way = fastq_2way
  )

  # Attempt sequence reconstruction of each cluster
  reconstructions, base_identity_counts, reconstruction_depths = cluster_sequence_reconstruction(
      reads,
      cluster_reads,
      fastq_2way = fastq_2way,
      mask = mask,
      sequencing_accuracy = sequencing_accuracy, 
      max_reconstruction_depth = max_reconstruction_depth,
      reconstruction_method = reconstruction_method,
      sample_id = sample_id,
      verbose = True
  )

  # Attempt to associate each cluster with one of the templates
  cluster_assignments = assign_clusters_to_templates(
      cluster_hashes, 
      templates, 
      template_hash_f_start = template_hash_f_start, hash_f_length = hash_f_length,
      template_hash_r_start = template_hash_r_start, hash_r_length = hash_r_length,
      max_hash_defect = max_hash_defect,
      sample_id = sample_id
  )
  template_counts = [len(cluster_reads[cluster_assignments.index(i)]) if i in cluster_assignments else 0 for i in range(len(templates))]
 
  # Save raw analysis results
  outpath = open('{}_data.p'.format(output_prefix), 'wb')
  output = {
    'sample_id': sample_id,
    'path_forward': path_forward,
    'path_reverse': path_reverse,
    'templates': templates,
    'cluster_hashes': cluster_hashes,
    'cluster_reads': cluster_reads,
    'cluster_sizes': [len(lst) for lst in cluster_reads],
    'cluster_sizes_UMI': cluster_sizes_UMI,
    'cluster_assignments': cluster_assignments,
    'cluster_reconstructions': reconstructions,
    'cluster_reconstruction_depths': reconstruction_depths,
    'cluster_base_counts': base_identity_counts,
    'template_counts': template_counts,
    'max_reads': max_reads,
    'num_reads': len(reads_raw),
    'num_reads_selfaligned': len(reads_selfaligned) if selfalign else None,
    'num_reads_alignleft': len(reads_aligned_left) if align_left_marker is not None else None,
    'num_reads_alignright': len(reads_aligned_right) if align_right_marker is not None else None,
    'max_hash_defect': max_hash_defect,
    'min_cluster_abundance': min_cluster_abundance,
    'sequencing_accuracy': sequencing_accuracy,
    'max_reconstruction_depth': max_reconstruction_depth,
    'reconstruction_method': reconstruction_method,
    'mask': mask,
    'hash_args': {
      'hash_f_start': hash_f_start,
      'hash_f_length': hash_f_length,
      'hash_r_start': hash_r_start,
      'hash_r_length': hash_r_length,
      'template_hash_f_start': template_hash_f_start,
      'template_hash_r_start': template_hash_r_start
    },
    'UMI_args': {
      'UMI_f_start': UMI_f_start,
      'UMI_f_length': UMI_f_length,
      'UMI_r_start': UMI_r_start,
      'UMI_r_length': UMI_r_length,
    },
    'alignment_args': {
      'selfalign': selfalign,
      'align_left_marker': align_left_marker,
      'align_left_position': align_left_position,
      'align_right_marker': align_right_marker,
      'align_right_position': align_right_position
    }
  }

  if full_output:
    output.update({
        'reads_raw': reads_raw,
        'reads_selfaligned': reads_selfaligned,
        'reads_leftaligned': reads_aligned_left,
        'reads_rightaligned': reads_aligned_right,
        'reads_masked': reads_masked
    })

  pickle.dump(output, outpath)
  outpath.close()

  # Make plots
  if sample_id is None:
    plot_title = 'Counts from {} reads'.format(len(reads_raw))
  else:
    plot_title = 'Counts from {} reads (SAMPLE {})'.format(len(reads_raw), sample_id)
  plot_counts_bar(templates, template_counts, output_prefix, title=plot_title)

  return output

def _call_func_starstarargs(args):
  func, kwargs = args
  return func(**kwargs)

def cluster_reads_by_hash(reads, hash_f_start = 35, hash_f_length = 24, hash_r_start = 36, hash_r_length = 24, fastq_2way = True, max_hash_defect = 5, max_reconstruction_depth = 1000, reconstruction_method = 'Levenshtein-median', min_cluster_size = 3, cluster_refresh_interval = 2500, sample_id = None, verbose = False):
  # Steps:
  # For each forward/reverse read, do the following:
  #  1) Identify the forward hash, which is located at forward_read[23:47]
  #  2) Identify the reverse hash, which is located at reverse_read[24:48]
  #  3) Concatenate the forward/reverse hashes to get an overall hash. Determine if this hash matches a previously observed hash
  #    a) if not, add to a list of hashes.
  #    b) if so, add the read name to a list of reads associated with that cluster
  #  4) Filter clusters that are smaller than the minimum cluster size
  # Return two lists:
  #  - a list of the hashes matched to each cluster
  #  - a list of the reads matched to each cluster

  # An issue with this approach is that some clusters may merge
  # as more sequences are processed. Thus, we periodically check
  # the cluster consensus hashes and combine clusters within
  # the allowed hash defect. We also clean up the cluster consensus
  # sequences at this time.

  # After an initial pass through the dataset to determine the clusters,
  # the dataset is processed again with fixed clusters to perform
  # a final assignment of reads to each cluster.

  # We maintain the following information throughout
  #  - a dict mapping hash subsequences to a "cluster index"
  #  - the sets of hash sequences associated with each cluster
  #  - the sets of read indices associated with each cluster

  def _merge_similar_clusters(num_clusters, cluster_sequences, cluster_reads):
    # remap cluster indices to a single cluster index if the consensus sequences are similar enough
    # this is done by finding connected components of the graph formed with edges between clusters that are close
    adj_matrix = np.zeros((num_clusters, num_clusters), dtype=np.int)
    for c1_idx, c1_seq in enumerate(tqdm.tqdm(cluster_sequences, desc='Cluster merge (1/2)...', disable=not verbose, leave=False)):
      adj_matrix[c1_idx, :] = [1 if utils.levenshtein_distance(c1_seq, c2_seq, score_cutoff=max_hash_defect+1)<=max_hash_defect else 0 for c2_seq in cluster_sequences]

    unprocessed_cluster_idxs = set(range(num_clusters))
    new_clusters = {}
    num_new_clusters = 0
    pbar = tqdm.tqdm(desc='Cluster merge (2/2)...', total=num_clusters, disable=not verbose, leave=False)
    while len(unprocessed_cluster_idxs) > 0:
      c_idx = next(iter(unprocessed_cluster_idxs))

      to_process = set([c_idx])
      while len(to_process) > 0:
        c_i = to_process.pop()

        if c_i not in unprocessed_cluster_idxs:  continue
        
        new_clusters[c_i] = num_new_clusters
        
        to_process.update(np.argwhere(adj_matrix[c_i,:]).flatten())
        unprocessed_cluster_idxs.remove(c_i)
        pbar.update(1)

      num_new_clusters += 1

    # remake/update cluster info
    cluster_reads_new = [[] for _ in range(num_new_clusters)]
    for c_idx in new_clusters:
      c_idx_new = new_clusters[c_idx]
      cluster_reads_new[c_idx_new].extend(cluster_reads[c_idx])
    cluster_reads_new = [list(set(r)) for r in cluster_reads_new]

    cluster_hashes_new = [[read_hashes[idx] for idx in c_reads] for c_reads in cluster_reads_new]
    cluster_sequences_new = [utils.consensus_sequence(c_hashes[:max_reconstruction_depth], method=reconstruction_method) for c_hashes in cluster_hashes_new]

    hash_subseqs_to_cluster = {}
    for c_idx,h in enumerate(cluster_sequences_new):
      if h is None:  continue

      for offset in range(0, len(h) - hash_subseq_length + 1, hash_subseq_length):
        h_sub = h[offset:offset+hash_subseq_length]
        hash_subseqs_to_cluster[h_sub] = hash_subseqs_to_cluster.get(h_sub, set()) | set([c_idx])

    return num_new_clusters, cluster_sequences_new, cluster_reads_new, cluster_hashes_new, hash_subseqs_to_cluster

  num_clusters = 0
  hash_subseqs_to_cluster = {}
  cluster_hashes = []
  cluster_reads = []
  cluster_sequences = []

  hash_length = hash_f_length + hash_r_length

  hash_subseq_length = hash_length//(max_hash_defect+1)

  read_hashes = [read_to_hash(read_seq, hash_f_start = hash_f_start, hash_f_length = hash_f_length, hash_r_start = hash_r_start, hash_r_length = hash_r_length, fastq_2way = fastq_2way) for _, read_seq in tqdm.tqdm(reads, desc='Hashing reads', disable = not verbose, leave=False)]
  
  for read_idx, hash_full in enumerate(tqdm.tqdm(read_hashes, desc='Clustering reads', disable=not verbose)):
    # extract hash; hash_f/hash_r starts at 23/24, but UMIs add 12nts to each end
    if len(hash_full) != hash_length:  continue

    # check if we've observed this hash before, or one with up to <max_hash_defect> mutations
    potential_clusters = set(
        idx
        for i in range(max_hash_defect+1)
        for idx in hash_subseqs_to_cluster.get(hash_full[hash_subseq_length*i:hash_subseq_length*(i+1)], [])
    )
    min_dist = np.inf
    clusters = set()
    for idx in potential_clusters:
      seq = cluster_sequences[idx]
      dist = utils.levenshtein_distance(seq, hash_full)
      if dist <= max_hash_defect:
        clusters.add(idx)

    # no cluster was matched, make a new (blank) cluster
    if len(clusters) == 0:
      clusters = set([num_clusters])
      cluster_hashes.append([])
      cluster_reads.append([])
      cluster_sequences.append('N'*hash_length)
      num_clusters += 1

    # associate each substring of the hash with each matching cluster
    for offset in range(0, len(hash_full) - hash_subseq_length + 1, hash_subseq_length):
      hash_subseq = hash_full[offset:offset+hash_subseq_length]
      hash_subseqs_to_cluster[hash_subseq] = hash_subseqs_to_cluster.get(hash_subseq, set()) | clusters

    # update cluster information for each matching cluster
    for idx in clusters:
      cluster_hashes[idx].append(hash_full)
      cluster_reads[idx].append(read_idx)
      cluster_sequences[idx] = utils.consensus_sequence(cluster_hashes[idx][:max_reconstruction_depth], method=reconstruction_method)

    # If we've reached the cluster refresh interval, merge similar clusters
    if read_idx % cluster_refresh_interval == cluster_refresh_interval-1 or read_idx == len(reads)-1:
      num_clusters, cluster_sequences, cluster_reads, cluster_hashes, hash_subseqs_to_cluster = _merge_similar_clusters(num_clusters, cluster_sequences, cluster_reads)

  # Perform phase 2 of the clustering, which is an iterative process of
  #   1) Reassign reads to their closest cluster, if it is within the max_hash_defect
  #   2) Recompute cluster hash sequence with the specified reconstruction method
  #   3) Merge clusters within the max_hash_defect
  # This is repeated until no clusters are merged in the final step, for a max of 5 times

  for phase2_iters in tqdm.trange(20, desc='Phase 2: Cluster refinement', disable=not verbose):
    # with clusters fixed, reassign reads to their closest clusters
    cluster_reads = [[] for _ in range(len(cluster_sequences))]
  
    with mp.Pool(20) as p:
      args_lst = [(_find_closest_cluster, {
          'hash': hash,
          'cluster_sequences': cluster_sequences,
          'hash_subseqs_to_cluster': hash_subseqs_to_cluster,
          'correct_hash_length': hash_length,
          'max_hash_defect': max_hash_defect,
          'hash_subseq_length': hash_subseq_length
      }) for hash in read_hashes]
      res_iter = p.imap(_call_func_starstarargs, args_lst, chunksize=100)
      for read_idx, res in enumerate(tqdm.tqdm(res_iter, total=len(args_lst), desc='Cluster assignment', disable=not verbose, leave=False)):
        cluster_idx, cluster_dist = res
        if cluster_dist is not None:
          cluster_reads[cluster_idx].append(read_idx)
  
    # recompute cluster hashes based on new read assignments
    cluster_hashes = [[read_hashes[idx] for idx in c_reads] for c_reads in cluster_reads]
    cluster_sequences = [utils.consensus_sequence(c_hashes[:max_reconstruction_depth], method=reconstruction_method) if len(c_hashes)>0 else '' for c_hashes in cluster_hashes]
  
    # cluster merging
    num_clusters_old = num_clusters
    num_clusters, cluster_sequences, cluster_reads, cluster_hashes, hash_subseqs_to_cluster = _merge_similar_clusters(num_clusters, cluster_sequences, cluster_reads)

    if num_clusters_old == num_clusters:
      break

  # filter clusters that are too small
  cluster_sequences_filt, cluster_reads_filt = [],[]
  for hash_seq, read_idxs in zip(tqdm.tqdm(cluster_sequences, desc='Cluster filtering and final reconstruction', disable=not verbose), cluster_reads):
    if len(read_idxs) >= min_cluster_size:
      cluster_sequences_filt.append(hash_seq)
      cluster_reads_filt.append(read_idxs)


  if len(cluster_sequences_filt) > 0:
    return zip(*sorted(zip(cluster_sequences_filt, cluster_reads_filt), key=lambda x: len(x[1]), reverse=True))
  else:
    return [],[]

def _find_closest_cluster(hash, cluster_sequences, hash_subseqs_to_cluster, correct_hash_length, max_hash_defect, hash_subseq_length):
    # get potential clusters that match below the max_hash_defect
    potential_clusters = list(set(
        idx
        for i in range(max_hash_defect+1)
        for idx in hash_subseqs_to_cluster.get(hash[hash_subseq_length*i:hash_subseq_length*(i+1)], [])
    ))
    if len(potential_clusters) == 0:  return None, None

    # choose the closest cluster
    dists = [utils.levenshtein_distance(hash, cluster_sequences[cluster_idx]) for cluster_idx in potential_clusters]
    argmin_dists = np.argmin(dists)
    if dists[argmin_dists] <= max_hash_defect:
      return potential_clusters[argmin_dists], dists[argmin_dists]
    else:
      return None, None

def calc_cluster_sizes_UMI(reads, cluster_read_indices, UMI_f_start = 0, UMI_f_length = 12, UMI_r_start = 0, UMI_r_length = 12, fastq_2way = True):
  # conflates reads that have UMIs that are different by up to 1 mutation

  cluster_sizes_UMI = []

  for read_idxs in cluster_read_indices:
    if len(read_idxs) == 0:
      cluster_sizes_UMI.append(0)
      continue

    UMI_counts = {}
    for read_idx in read_idxs:
      _, seq = reads[read_idx]
      UMI = read_to_UMI(seq, UMI_f_start = UMI_f_start, UMI_f_length = UMI_f_length, UMI_r_start = UMI_r_start, UMI_r_length = UMI_r_length, fastq_2way = fastq_2way)
      UMI_counts[UMI] = UMI_counts.get(UMI, 0) + 1

    UMIs_sorted,_ = zip(*sorted(UMI_counts.items(), key = lambda x: x[1], reverse=True))

    UMIs_filt = set()
    UMIs_used = set()
    for UMI in UMIs_sorted:
      if UMI not in UMIs_used:
        UMIs_filt.add(UMI)
        UMIs_used.add(UMI)
        for i in range(len(UMI)):
          for nt in 'ATCG':
            UMIs_used.add(UMI[:i] + nt + UMI[i+1:])

    cluster_sizes_UMI.append(len(UMIs_filt))

  return cluster_sizes_UMI

def cluster_sequence_reconstruction(reads, cluster_read_indices, fastq_2way = True, sequencing_accuracy = 0.75, mask = None, max_reconstruction_depth = 100, reconstruction_method = 'Levenshtein-median', sample_id = None, verbose = False):
  # For each cluster:
  #   Output the reads in the cluster to a temporary .fasta file
  #     If reads are 2-way, first do a self-alignment
  #     If there are more than <max_reconstruction_depth> reads, use the first <max_reconstruction_depth> reads.
  #   Run the specified reconstruction method
  reconstruction_depths = []
  base_identity_counts_all = []
  consensus_sequences = []
  for read_idxs in tqdm.tqdm(cluster_read_indices, desc='Reconstructing cluster sequences', disable=not verbose):
    # Perform self-alignment, if necessary. Then pare down to the first <max_reconstruction_depth> reads
    if fastq_2way:
      cluster_reads = [utils.fastq_2way_selfalign(reads[idx], acc_cutoff = sequencing_accuracy, align_s2_length = 20, min_alignment_overlap = 10) for idx in read_idxs]
    else:
      cluster_reads = [reads[idx] for idx in read_idxs]
    cluster_reads = [read for read in cluster_reads if read[1] is not None][:max_reconstruction_depth]

    if mask is not None:
      cluster_reads = [(name, ''.join(nt if mask_bit else 'N' for nt, mask_bit in zip(seq, mask))) for name,seq in cluster_reads]
    _, sequences = zip(*cluster_reads)

    consensus, base_identity_counts = utils.consensus_sequence(sequences, method=reconstruction_method)

    consensus_sequences.append(consensus)
    base_identity_counts_all.append(base_identity_counts)
    reconstruction_depths.apend(len(cluster_reads))

  return consensus_sequences, base_identity_counts_all, reconstruction_depths

def assign_clusters_to_templates(cluster_hashes, templates, template_hash_f_start = 23, hash_f_length = 24, template_hash_r_start = 24, hash_r_length = 24, max_hash_defect = 3, sample_id = None, verbose = True):
  # Extract hashes for each template:
  template_hashes = []
  for _, template_seq in templates:
    template_hashes.append(sequence_to_hash(
        template_seq, 
        hash_f_start = template_hash_f_start, hash_f_length = hash_f_length,
        hash_r_start = template_hash_r_start, hash_r_length = hash_r_length,
    ))

  # For each cluster, see if one of the template hashes is included in the cluster's list of hashes
  cluster_assignments = []
  for h in cluster_hashes:
    dists = [utils.levenshtein_distance(t_hash, h) for t_hash in template_hashes]
    min_dist = min(dists)
    if min_dist <= max_hash_defect:
      cluster_assignments.append(dists.index(min_dist))
    else:
      cluster_assignments.append(None)

  return cluster_assignments

def plot_counts_bar(templates, counts, output_prefix, title = None):

  bar_x = np.arange(len(templates))
  bar_heights = counts
  bar_width = .9
  bar_tick_labels = [name for name,_ in templates]
  ylims = (0, max(counts)*1.02+.01)

  plt.figure(figsize = (.32*len(templates),4.8))
  plt.bar(x = bar_x, height = bar_heights, width = bar_width, data = counts)
  plt.ylabel('Counts')
  plt.xticks(bar_x, bar_tick_labels, rotation='vertical')

  for x, c in zip(bar_x, counts):
    plt.text(x, ylims[1]/50., str(c), horizontalalignment = 'center', verticalalignment = 'bottom', rotation='vertical')

  plt.ylim(ylims)

  if title is not None:
    plt.title(title)

  plt.savefig('{}_counts.pdf'.format(output_prefix), bbox_inches='tight')


def read_to_hash(seq, hash_f_start = 23, hash_f_length = 24, hash_r_start = 24, hash_r_length = 24, fastq_2way = True):
  # Forward hash is given by:
  #  seq_f[hash_f_start:hash_f_start+hash_f_length] 
  # Reverse hash is given by:
  #  seq_r[hash_r_start:hash_r_start+hash_r_length]
  # Full hash is:
  #  forward_hash + complement(reverse_hash)
  if fastq_2way:
    seq_f, seq_r = seq
    hash_f = seq_f[hash_f_start:hash_f_start+hash_f_length]
    hash_r = seq_r[hash_r_start:hash_r_start+hash_r_length]
    return hash_f + utils.sequence_complement(hash_r)
  else:
    hash_f = seq[hash_f_start:hash_f_start+hash_f_length]
    hash_r = seq[len(seq)-hash_r_start-1:len(seq)-hash_r_start-1+hash_r_length]
    return hash_f + hash_r

def sequence_to_hash(sequence, hash_f_start = 23, hash_f_length = 24, hash_r_start = 24, hash_r_length = 24):
  # Parameters are the same as for read_to_hash
  return read_to_hash(sequence, hash_f_start, hash_f_length, hash_r_start, hash_r_length, fastq_2way = False)

def read_to_UMI(seq, UMI_f_start = 0, UMI_f_length = 12, UMI_r_start = 0, UMI_r_length = 12, fastq_2way = True):
  # Parameters are interpreted as in read_to_hash()
  # but default UMI lengths are 12 for both forward and reverse
  if fastq_2way:
    seq_f, seq_r = seq
    UMI_f = seq_f[UMI_f_start:UMI_f_start+UMI_f_length]
    UMI_r = seq_r[UMI_r_start:UMI_r_start+UMI_r_length]
    return UMI_f + utils.sequence_complement(UMI_r)
  else:
    UMI_f = seq[UMI_f_start:UMI_f_start+UMI_f_length]
    UMI_r = seq[len(seq)-UMI_r_start-1:len(seq)-UMI_r_start-1+UMI_r_length]
    return UMI_f + UMI_r


