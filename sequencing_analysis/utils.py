import subprocess
import os
import random
import itertools as it
import copy

import Bio.SeqRecord, Bio.Seq, Bio.SeqIO, Bio.Align.Applications, Bio.Align.AlignInfo
import Levenshtein
import tqdm
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def import_fasta(path, sample_id = None, verbose = False):
  file = open(path)

  sequences = []

  pbar_desc = f'Loading FASTA: {path}'
  if sample_id is not None:
    pbar_desc = f'SAMPLE {sample_id}: {pbar_desc}'
  pbar = tqdm.tqdm(desc=pbar_desc, unit=' seqs', disable=not verbose, leave=False)

  cur_name, cur_sequence = '',''
  for line in file:
    if line[0]=='>':
      if cur_sequence != '':
        sequences.append((cur_name, cur_sequence))
        pbar.update(1)
      cur_name = line[1:].strip()
      cur_sequence = ''
    else:
      cur_sequence += line.strip()
  sequences.append((cur_name, cur_sequence))

  pbar.close()
  file.close()

  return sequences

def export_fasta(path, sequences, sample_id = None, verbose = False):
  file = open(path, 'w')

  pbar_desc = f'Exporting FASTA: {path}'
  if sample_id is not None:
    pbar_desc = f'SAMPLE {sample_id}: {pbar_desc}'
  pbar = tqdm.tqdm(desc=pbar_desc, unit=' seqs', disable=not verbose, leave=False)

  for name, seq in sequences:
    file.write(f'>{name}\n')
    file.write(f'{seq}\n')
    pbar.update(1)

  pbar.close()

  file.close()

def import_fastq(path, max_reads = None, sample_id = None, max_sequence_length = None, sequence_pretruncate = 0, verbose = False):
  if max_reads is None:  max_reads = float('inf')

  file = open(path)

  sequences = []

  if sample_id is None:
    pbar_desc = f'Loading FASTQ: {path}'
  else:
    pbar_desc = f'SAMPLE {sample_id}: Loading FASTQ: {path}'
  pbar = tqdm.tqdm(desc=pbar_desc, unit=' reads', dynamic_ncols = True, disable = not verbose)

  lines = [file.readline() for _ in range(4)]
  while lines[-1] != '' and len(sequences) < max_reads:
    cur_name = lines[0].split(" ")[0]
    cur_sequence = lines[1].strip()
    cur_sequence = cur_sequence[sequence_pretruncate:]
    if max_sequence_length is not None:
      cur_sequence = cur_sequence[:max_sequence_length]

    sequences.append((cur_name, cur_sequence))
    pbar.update(1)

    lines = [file.readline() for _ in range(4)]

  pbar.close()

  return sequences

def import_fastq_2way(path_f, path_r, max_reads = None, sample_id = None, max_sequence_length = None, sequence_pretruncate = 0, align = False, verbose = False):
  reads_forward = import_fastq(path_f, max_reads = max_reads, sample_id = sample_id, max_sequence_length = max_sequence_length, sequence_pretruncate = sequence_pretruncate, verbose = verbose)
  reads_reverse = import_fastq(path_r, max_reads = max_reads, sample_id = sample_id, max_sequence_length = max_sequence_length, sequence_pretruncate = sequence_pretruncate, verbose = verbose)
  
  # Combine forward and reverse reads into the same list
  reads_all = []
  for i, ((n1, s1), (n2, s2)) in enumerate(zip(tqdm.tqdm(reads_forward, desc='Checking 2-way FASTQ reads...', disable=not verbose, leave=False), reads_reverse)):
    if n1 != n2:
      print("\nWarning: Inconsistent forward/reverse read {} in {} and {}".format(i, path_f, path_r))
      continue

    reads_all.append((n1, (s1, s2)))

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

  (align_idx_start, align_idx_end), align_score = align_sequences_hamming(s2_c_sub, s1)
  
  overlap_len = min(align_s1_length-max(align_idx_start,0), align_s2_length-max(-align_idx_start,0))
  if overlap_len >= min_alignment_overlap and align_score >= overlap_len*acc_cutoff:
    s_align = ''

    if align_idx_start > 0:
      s_align += s1[:align_idx_start]

    s_align += ''.join(
        nt1 if nt1==nt2 or nt2=='N' else (nt2 if nt1=='N' else 'N')
        for nt1,nt2 in zip(s1[max(align_idx_start,0):], s2_c[max(-align_idx_start,0):])
    )

    if align_idx_start > 0:
      s_align += s2_c[-align_idx_start:]
      
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
  
  (align_idx_start, align_idx_end), align_score = align_sequences_levenshtein(marker, s)
  if align_score >= len(marker)*acc_cutoff:
    if side == 'left':
      return fastq_offset(read, position + len(marker) - align_idx_end, side = side)
    elif side == 'right':
      return fastq_offset(read, (len(s) - align_idx_start) - position, side = side)
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
    (align_idx_start, align_idx_end), align_score = align_sequences_levenshtein(orient, s)
    if align_score >= len(orient)*acc_cutoff:
      return (n,s)
    
    (r_align_idx_start, r_align_idx_end), r_align_score = align_sequences_levenshtein(sequence_complement(orient), s)
    if align_score >= len(orient)*acc_cutoff:
      return (n+'_C',sequence_complement(s))

  return (n,None)

complements = {'A': 'T', 'T':'A', 'G':'C', 'C':'G', '*':'*', 'N':'N'}
def sequence_complement(seq, memo = {}):
  if seq not in memo:
    memo[seq] = ''.join([complements[nt] for nt in reversed(seq)])
    
  return memo[seq]
 
def align_sequences(template, sequence):
  return align_sequences_levenshtein(template, sequence)

def align_sequences_levenshtein(template, sequence):
  # The best position is given by levenshtein distance, but without penalizing extra nucleotides in the sequence
  seq_len = len(sequence)
  tpt_len = len(template)

  aligner=Bio.Align.PairwiseAligner(mode='global', match_score=0, mismatch_score=-1, open_gap_score=-1, extend_gap_score=-1, target_end_gap_score=0, query_end_gap_score=-1)

  alignments = aligner.align(template, sequence)
  alignment = alignments[0]
  alignment_seq_blocks = alignment.aligned[1]
  min_seq_idx = min([blk[0] for blk in alignment_seq_blocks])
  max_seq_idx = max([blk[1] for blk in alignment_seq_blocks])
  return (min_seq_idx, max_seq_idx), tpt_len + alignment.score
  

def align_sequences_hamming(template, sequence):
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
      best_pos = i
      best_score = score
  for i in range(len(template)):
    template_sub = template[i:i+seq_len]
    sequence_sub = sequence[:len(template_sub)]
    score = sum(a==b for a,b in zip(sequence_sub,template_sub))
    if score >= best_score:
      best_pos = -i
      best_score = score
  return (best_pos, best_pos + len(template)), best_score

def consensus_sequence(sequences, method = 'Levenshtein-median'):
  if method = 'naive':
    return consensus_sequence_naive(sequences)
  elif reconstruction_method = 'ClustalOmega':
    return consensus_sequence_ClustalOmega(sequences)
  else:
    return consensus_sequence_Levenshtein_median(sequences)

def consensus_sequence_naive(sequences):
  max_seq_len = max([len(seq) for seq in sequences])

  base_identity_counts = np.zeros((max_seq_len, 4))

  for seq in sequences:
    for i,nt in enumerate(seq):
      if nt in 'ATCG':  
        base_identity_counts[i, 'ATCG'.index(nt)] += 1
      else:
        base_identity_counts[i, :] += .25

    consensus_sequence = ''
    max_nts = np.argmax(base_identity_counts, axis=1)
    nt_cts = np.sum(base_identity_counts, axis=1)
    for nt_idx, max_nt in enumerate(max_nts):
      if base_identity_counts[nt_idx,max_nt] > len(reads)/2:
        consensus_sequence += 'ATCG'[max_nt]
      elif nt_cts[nt_idx] >= len(reads)/2:  # check that a majority of strands were this long
        consensus_sequence += 'N'
      else:
        break

  return consensus_sequence, base_identity_counts

def consensus_sequence_ClustalOmega(sequences):
  # Escape clause if there's one or fewer reads
  if len(sequences) == 1:
    return sequences[0], None
  elif len(sequences) == 0:
    return None, None

  # Initialize temporary files for use with Clustal Omega
  infile, inpath = tempfile.mkstemp(suffix='.fasta', text=False)
  outfile, outpath = tempfile.mkstemp(suffix='.fasta', text=False)
  os.close(infile)
  os.close(outfile)

  # Export to a .fasta file (required input format for Clustal Omega)
  reads = [(f'read_{i}', seq) for i,seq in enumerate(sequences)]
  utils.export_fasta(inpath, reads, verbose=verbose)

  # Run ClustalOmega
  clustalomega_cline = Bio.Align.Applications.ClustalOmegaCommandline(infile = inpath, outfile = outpath, auto=True, force=True)
  stdout, stderr = clustalomega_cline()

  # Parse output
  # The most natural way is with Bio.AlignIO but I can't import this bc we are missing sqlite3
  # So we have to make the MultipleSeqAlignment object manually
  alignment_seqs = utils.import_fasta(outpath, verbose=verbose)
  alignment_seqs_Bio = [Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq), id=name) for name,seq in alignment_seqs]
  alignment = Bio.Align.MultipleSeqAlignment(alignment_seqs_Bio)

  # Cleanup temporary files
  os.unlink(inpath)
  os.unlink(outpath)

  # Calculate consensus sequence using the "dumb_consensus()" method
  align_summary = Bio.Align.AlignInfo.SummaryInfo(alignment)
  consensus = align_summary.dumb_consensus(threshold=.5, ambiguous='N')
  pssm = align_summary.pos_specific_score_matrix(consensus)

  return consensus, pssm

def consensus_sequence_Levenshtein_median(sequences):
  if len(sequences) > 1:
    return Levenshtein.median(sequences)
  elif len(sequences) == 1:
    return sequences[0]
  else:  # sequences == []
    return None

def hamming_distance(seq1, seq2):
  return Levenshtein.hamming(seq1, seq2)

def levenshtein_distance(seq1, seq2, score_cutoff=None):
  # Note: the version of Levenshtein that I can get installed doesn't support score_cutoff
  return Levenshtein.distance(seq1, seq2)
