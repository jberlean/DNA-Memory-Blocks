import subprocess
import os
import random
import itertools as it
import copy

import tqdm
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def import_fasta(path, pbar = None, sample_id = None):
  file = open(path)

  sequences = []

  pbar_desc = 'Loading FASTA: {}'.format(path)
  if sample_id is not None:
    pbar_desc = 'SAMPLE {}: '.format(sample_id) + pbar_desc
  if pbar is None:
    pass
  else:
    pbar.set_description(pbar_desc)
    pbar.unit = ' seqs'
#    pbar.dynamic_ncols=True
    pbar.reset()
    close_pbar = False

#  print "Loading sequences from {}...\r".format(path), 

  cur_name, cur_sequence = '',''
  for line in file:
    if line[0]=='>':
      if cur_sequence != '':
        sequences.append((cur_name, cur_sequence))
#        print "Loading sequences from {}: {}\r".format(path, len(sequences)),
        if pbar is not None:
          pbar.update(1)
      cur_name = line[1:].strip()
      cur_sequence = ''
    else:
      cur_sequence += line.strip()
  sequences.append((cur_name, cur_sequence))

  if pbar is not None and close_pbar:
    pbar.close()

  return sequences


def import_fastq(path, max_reads = None, pbar = None, sample_id = None, max_sequence_length = None, sequence_pretruncate = 0):
  if max_reads is None:  max_reads = float('inf')

  file = open(path)

  sequences = []

  pbar_desc = 'Loading FASTQ: {}'.format(path)
  if sample_id is not None:
    pbar_desc = 'SAMPLE {}: '.format(sample_id) + pbar_desc
  if pbar is None:
#    pbar = tqdm.tqdm(desc=pbar_desc, unit=' reads', dynamic_ncols = True)
#    close_pbar = True
    pass
  else:
    pbar.set_description(pbar_desc)
    pbar.unit = ' reads'
#    pbar.dynamic_ncols = True
    pbar.reset()
    close_pbar = False
#  print "Loading sequences from {}...\r".format(path), 

  lines = [file.readline() for _ in range(4)]
  while lines[-1] != '' and len(sequences) < max_reads:
    cur_name = lines[0].split(" ")[0]
    cur_sequence = lines[1].strip()
    cur_sequence = cur_sequence[sequence_pretruncate:]
    if max_sequence_length is not None:
      cur_sequence = cur_sequence[:max_sequence_length]

    sequences.append((cur_name, cur_sequence))
#    print "Loading sequences from {} ({})...\r".format(path, len(sequences)), 
    if pbar is not None:
      pbar.update(1)

    lines = [file.readline() for _ in range(4)]

#  print "Loading sequences from {} ({})... Done!".format(path, len(sequences)) 
  if pbar is not None and close_pbar:
    pbar.close()
  return sequences

def import_fastq_2way(path_f, path_r, max_reads = None, pbar = None, sample_id = None, max_sequence_length = None, sequence_pretruncate = 0, align = False):
  reads_forward = import_fastq(path_f, max_reads = max_reads, pbar = pbar, sample_id = sample_id, max_sequence_length = max_sequence_length, sequence_pretruncate = sequence_pretruncate)
  reads_reverse = import_fastq(path_r, max_reads = max_reads, pbar = pbar, sample_id = sample_id, max_sequence_length = max_sequence_length, sequence_pretruncate = sequence_pretruncate)
  
  # Combine forward and reverse reads into the same list
  reads_all = []
#  seqs_f_np = np.zeros((len(reads_forward), READ_LENGTH), dtype=np.int)
#  seqs_r_c_np = np.zeros((len(reads_reverse), READ_LENGTH), dtype=np.int)
  for i, ((n1, s1), (n2, s2)) in enumerate(zip(reads_forward, reads_reverse)):
    if n1 != n2:
      print("\nWarning: Inconsistent forward/reverse read {} in {} and {}".format(i, path_f, path_r))
      continue

    reads_all.append((n1, (s1, s2)))
#    seqs_f_np[i,:] = [ord(c) for c in s1]
#    seqs_r_c_np[i,:] = [ord(c) for c in utils.sequence_complement(s2)]


  return reads_all

def fastq_2way_selfalign(read, acc_cutoff = .75, align_s1_length = None, align_s2_length = None, min_alignment_overlap = 20):
  n, (s1,s2) = read
  
  if align_s1_length is None:
    align_s1_length = len(s1)
  if align_s2_length is None:
    align_s2_length = len(s2)

  s2_c = sequence_complement(s2)
  s1_sub = s1[-align_s1_length:]
  s2_c_sub = sequence_complement(s2[-align_s2_length:])

  align_pos, align_score = align_sequences(s2_c_sub, s1)
  
  overlap_len = min(align_s1_length-max(align_pos,0), align_s2_length-max(-align_pos,0))
  if overlap_len >= min_alignment_overlap and align_score >= overlap_len*acc_cutoff:
    s_align = ''

    if align_pos > 0:
      s_align += s1[:align_pos]

    s_align += ''.join(
        nt1 if nt1==nt2 or nt2=='N' else (nt2 if nt1=='N' else 'N')
        for nt1,nt2 in zip(s1[max(align_pos,0):], s2_c[max(-align_pos,0):])
    )

    if align_pos > 0:
      s_align += s2_c[-align_pos:]
      
    return (n, s_align)
  else:
    return (n, None)

def fastq_offset(read, offset, side = 'left'):
  # Shifts the read sequence by `offset` nucleotides
  # where `offset` > 0 indicates a shift to the right and `offset` < 0 indicates a shift to the left.
  # The value of `side` indicates whether the offset is measured from the left or right end of the sequence.
  # The behavior of fastq_offset() is as follows:
  #   side = 'left':
  #     offset >= 0:
  #       seq_new = 'N'*offset + seq_old
  #     offset < 0:
  #       seq_new = seq_old[-offset:]
  #   side = 'right':
  #     offset >= 0:
  #       seq_new = seq_old[:-offset]
  #     offset < 0:
  #       seq_new = seq_old + 'N'*(-offset)
  n,s = read

  if side == 'left' and offset > 0:
    s_offset = 'N'*offset + s
  elif side == 'left' and offset <= 0:
    s_offset = s[-offset:]
  elif side == 'right' and offset > 0:
    s_offset = s[:-offset]
  elif side == 'right' and offset <= 0:
    s_offset = s + 'N'*(-offset)
  else:
    s_offset = None

  if offset != 0:
    n_offset = f'{n}_{side}{offset}'
  else:
    n_offset = n

  return (n_offset, s_offset)
  
def fastq_2way_align_to_marker(read, marker, position, side = 'left', acc_cutoff = .75, orient=False):
  n,(sf,sr) = read

  if side == 'left':
    n_align, sf_align = fastq_align_to_marker((n,sf), marker, position, side=side, acc_cutoff=acc_cutoff, orient=False)
    s_align = None if sf_align is None else (sf_align, sr)
  else:
    n_align, sr_align_C = fastq_align_to_marker((n,sequence_complement(sr)), marker, position, side=side, acc_cutoff=acc_cutoff, orient=False)
    s_align = None if sr_align_C is None else (sf, sequence_complement(sr_align_C))

  if s_align is None and orient:
    n_align, s_align = fastq_2way_align_to_marker((n+'_C', (sr,sf)), marker, position, side=side, acc_cutoff=acc_cutoff, orient=False)
  
  if s_align is None:
    return (n, None)
  else:
    return (n_align, s_align)

def fastq_align_to_marker(read, marker, position, side = 'left', acc_cutoff = .75, orient=False):
  # If side == 'left', then position is measured from the left end.
  # If side == 'right', then position is measured from the right end.

  n,s = read
  
  align_pos, align_score = align_sequences(marker, s)
  if align_score >= len(marker)*acc_cutoff:
    if side == 'left':
      return fastq_offset(read, position - align_pos, side = side)
    elif side == 'right':
      return fastq_offset(read, (len(s) - align_pos) - position, side = side)
  elif orient:
    n_align, s_align = fastq_align_to_marker((n+'_C', sequence_complement(s)), marker, position, side = side, acc_cutoff=acc_cutoff, orient=False)
    if s_align is not None:
      return (n_align, s_align)
    else:
      return (n, None)
  else:
    return (n, None)

def fastq_orient(read, orient_markers, acc_cutoff = .75):
  n,s = read
  for orient in orient_markers:
    align_pos, align_score = align_sequences(orient, s)
    print(orient, align_pos, align_score)
    if align_score >= len(orient)*acc_cutoff:
      return (n,s)
    
    r_align_pos, r_align_score = align_sequences(sequence_complement(orient), s)
    print('*',orient, r_align_pos, r_align_score)
    if align_score >= len(orient)*acc_cutoff:
      return (n+'_C',sequence_complement(s))

  return (n,None)

complements = {'A': 'T', 'T':'A', 'G':'C', 'C':'G', '*':'*', 'N':'N'}
def sequence_complement(seq, memo = {}):
  if seq not in memo:
    memo[seq] = ''.join([complements[nt] for nt in reversed(seq)])
    
  return memo[seq]
 
#def align_sequences(sequence, template):
#  # Returns the optimal sequence alignment position assuming no deletions
#  # The best alignment is the one with the lowest hamming distance
#  seq_len = len(sequence)
#
#  best_pos = None
#  best_score = -1
#  for i in range(len(sequence)):
#    sequence_sub = sequence[i:]
#    template_sub = template[:seq_len-i]
#    score = sum(a==b for a,b in zip(sequence_sub,template_sub))
#    if score >= best_score:
#      best_pos = -i
#      best_score = score
#  for i in range(len(template)):
#    template_sub = template[i:i+seq_len]
#    sequence_sub = sequence[-len(template_sub):]  # this is wrong -- should be  sequence_sub = sequence[:len(template_sub)]
#    score = sum(a==b for a,b in zip(sequence_sub,template_sub))
##    print i, score, best_pos, best_score
##    print template_sub
##    print sequence
#    if score >= best_score:
#      best_pos = i
#      best_score = score
##  print best_pos, best_score
#  return best_pos, best_score

def align_sequences(sequence, template):
  # Returns the optimal sequence alignment position assuming no deletions
  # The best alignment is the one with the lowest hamming distance
  seq_len = len(sequence)

  best_pos = None
  best_score = -1
  for i in range(len(sequence)):
    sequence_sub = sequence[i:]
    template_sub = template[:seq_len-i]
    score = sum(a==b for a,b in zip(sequence_sub,template_sub))
    if score >= best_score:
      best_pos = -i
      best_score = score
  for i in range(len(template)):
    template_sub = template[i:i+seq_len]
    sequence_sub = sequence[:len(template_sub)]
    score = sum(a==b for a,b in zip(sequence_sub,template_sub))
#    print i, score, best_pos, best_score
#    print template_sub
#    print sequence
    if score >= best_score:
      best_pos = i
      best_score = score
#  print best_pos, best_score
  return best_pos, best_score

def hamming_distance(seq1, seq2):
  assert(len(seq1) == len(seq2))
  return sum(a!=b for a,b in zip(seq1,seq2))

def levenshtein_distance(seq1, seq2):
  dists = np.empty((len(seq1)+1, len(seq2)+1))
  dists[:len(seq1)+1,0] = np.arange(len(seq1)+1)
  dists[0,:len(seq2)+1] = np.arange(len(seq2)+1)
  for i in range(len(seq1) + len(seq2) + 1):
    max_r = min(i-1, len(seq1))
    min_r = max(1, i - len(seq2))
    for r in range(min_r, max_r+1):
      c = i - r
      diff = 0 if seq1[r-1]==seq2[c-1] else 1
      dists[r,c] = min(dists[r-1,c]+1, dists[r,c-1]+1, dists[r-1,c-1]+diff)

  return dists[-1,-1]

    
