from collections import OrderedDict
from collections import Counter, defaultdict
from functools import partial
import json
import h5py
import os
import random
from random import shuffle
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class OnVTGDataset(Dataset):

    def __init__(
        self,
        split,
        anno_file,
        vid_feat_dir,
        text_feat_dir,
        use_tokenizer,
        is_training,

        long_term_window_size,
        max_num_window,
        max_text_len,
        max_num_text,
        clip_size,
        clip_stride,
        downsample_rate,
        
        normalize_vid=False,
        normalize_text=False,
        drop_text=False,

        crop_ratio=(0.9, 1.0),
        trunc_thresh=0.5,

        num_epochs=1,
        hard_negative_prob=0.0,
        hard_negative_topk=32,
        hard_negative_mode='lexical',
        prefix_sampling_prob=0.0,
        min_prefix_len=32,
        prefix_state_probs=(0.35, 0.35, 0.30),
        video_cache_size=0,
        benchmark="standard",
    ):
        super(OnVTGDataset, self).__init__()

        assert os.path.exists(anno_file)
        if not use_tokenizer:
            assert text_feat_dir is not None, (
                "text features must be given if tokenizer is not specified"
            )
        assert isinstance(downsample_rate, int) and downsample_rate >= 1
        if crop_ratio is not None:
            assert isinstance(crop_ratio, (list, tuple))

        self.benchmark = benchmark
        if is_training and benchmark == "hi_activitynet":
            raise ValueError("HI-ActivityNet is evaluation-only")
        self.split = split
        self.is_training = is_training
        self.epoch = 0  # this must be updated upon starting a new epoch

        self.anno_file = anno_file
        self.vid_feat_dir = vid_feat_dir
        self.text_feat_dir = text_feat_dir
        if self.vid_feat_dir.endswith('.h5'):
            self.vid_h5 = None
        if self.text_feat_dir is not None and self.text_feat_dir.endswith('.h5'):
            self.text_h5 = None
        self.tokenizer = GloVeTokenizer() if use_tokenizer else None

        self.long_term_window_size = long_term_window_size
        self.max_num_window = max_num_window
        self.max_text_len = max_text_len
        self.max_num_text = max_num_text
        self.clip_size = clip_size
        self.clip_stride = clip_stride * downsample_rate
        self.downsample_rate = downsample_rate

        self.normalize_vid = normalize_vid
        self.normalize_text = normalize_text
        self.drop_text = drop_text

        self.crop_ratio = crop_ratio
        self.trunc_thresh = trunc_thresh

        self.vid_dict, self.text_dict = self._parse_annotations()
        self.num_epochs = num_epochs
        self.hard_negative_prob = float(hard_negative_prob) if is_training else 0.0
        self.hard_negative_topk = int(hard_negative_topk)
        self.hard_negative_mode = str(hard_negative_mode)
        if self.hard_negative_mode not in ('lexical', 'random'):
            raise ValueError("hard_negative_mode must be 'lexical' or 'random'")
        self.prefix_sampling_prob = float(prefix_sampling_prob) if is_training else 0.0
        self.min_prefix_len = int(min_prefix_len)
        self.prefix_state_probs = tuple(float(x) for x in prefix_state_probs)
        self.video_cache_size = int(video_cache_size)
        self._video_cache = OrderedDict()
        if len(self.prefix_state_probs) != 3:
            raise ValueError("prefix_state_probs must contain pre/active/post weights")
        if abs(sum(self.prefix_state_probs) - 1.0) > 1e-5:
            raise ValueError("prefix_state_probs must sum to 1")
        self._negative_inverted = None
        self._all_text_ids = tuple(self.text_dict.keys())
        if self.hard_negative_prob > 0 and self.hard_negative_mode == 'lexical':
            self._build_negative_index()

        if self.is_training:
            self.data_list = self._build_train_samples()
        else:
            assert self.num_epochs == 1
            self.data_list = self._build_eval_samples()

    def _parse_annotations(self):
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        if self.benchmark == 'hi_activitynet' and anno.get('benchmark') != 'HI-ActivityNet':
            raise ValueError('Supply converted official HI-ActivityNet annotations, not generated negatives')
        if self.is_training and anno.get('benchmark') == 'HI-ActivityNet':
            raise ValueError('Official HI-ActivityNet must not enter training')
        vid_dict, text_dict = OrderedDict(), OrderedDict()
        for key, value in anno[self.split].items():
            fps, num_frames = float(value['fps']), int(value['num_frames'])
            if 'duration' in value:
                duration = float(value['duration'])
            else:
                duration = num_frames / fps
            num_clips = (value['num_clips'] + self.downsample_rate - 1) // self.downsample_rate

            text_ids, segments, relevance = tuple(), tuple(), tuple()
            for s, pair in enumerate(value['annotations']):
                text = pair['sentence'].strip()
                text_id = pair.get('sentence_id', key + '_{:04d}'.format(s))

                is_relevant = bool(pair.get('is_relevant', True))
                if is_relevant:
                    start = max(float(pair['segment'][0]), 0)
                    end = min(float(pair['segment'][1]), duration)
                else:
                    start, end = -1.0, -1.0
                seg_len = end - start
                if is_relevant and seg_len <= 0:
                    continue
                segment = (start, end)
                segments += (segment, )
                if text_id in text_dict:
                    raise ValueError(f'Duplicate query ID: {text_id}')
                text_dict[text_id] = {
                    'text'      : text,
                    'segment'   : np.array(segment),
                    'text_idx'  : s,
                    'vid_id'    : key,
                    'is_relevant': is_relevant,
                    'official_query_id': pair.get('official_query_id', text_id),
                    'is_strong': pair.get('is_strong', False),
                }
                text_ids += (text_id, )
                relevance += (is_relevant, )
            
            if len(text_ids) == 0:
                continue

            vid_dict[key] = {
                'fps'       : fps,
                'num_frames': num_frames,
                'num_clips' : num_clips,
                'duration'  : duration,
                'text_ids'  : text_ids,
                'segments'  : np.array(segments),
                'relevance' : np.asarray(relevance, dtype=bool),
            }

        return vid_dict, text_dict

    @staticmethod
    def _content_tokens(text):
        tokens = [
            token.strip(".,!?;:'\"()[]{}-/").lower()
            for token in text.split()
        ]
        return tuple(token for token in tokens if len(token) >= 3)

    def _build_negative_index(self):
        """Build a compact inverted index for on-demand lexical negatives."""

        inverted = defaultdict(list)
        for text_id, item in self.text_dict.items():
            for token in set(self._content_tokens(item['text'])):
                inverted[token].append(text_id)
        max_posting = max(1024, self.hard_negative_topk * 32)
        self._negative_inverted = {
            token: tuple(ids) for token, ids in inverted.items()
            if len(ids) <= max_posting
        }

    def _random_cross_video_query(self, source_text_id):
        source_vid = self.text_dict[source_text_id]['vid_id']
        for _ in range(64):
            candidate = random.choice(self._all_text_ids)
            if self.text_dict[candidate]['vid_id'] != source_vid:
                return candidate
        return source_text_id

    def _sample_query_id(self, text_id):
        if self.hard_negative_prob <= 0 or random.random() >= self.hard_negative_prob:
            return text_id, True
        if self.hard_negative_mode == 'random':
            candidate = self._random_cross_video_query(text_id)
            return candidate, candidate == text_id
        if self._negative_inverted is None:
            self._build_negative_index()
        counts = Counter()
        for token in set(self._content_tokens(self.text_dict[text_id]['text'])):
            counts.update(self._negative_inverted.get(token, ()))
        source_vid = self.text_dict[text_id]['vid_id']
        candidates = [
            candidate for candidate, _ in counts.most_common(self.hard_negative_topk * 4)
            if self.text_dict[candidate]['vid_id'] != source_vid
        ][:self.hard_negative_topk]
        if not candidates:
            return text_id, True
        return random.choice(candidates), False

    def _sample_prefix_len(self, vid_len, local_segments):
        if self.prefix_sampling_prob <= 0 or random.random() >= self.prefix_sampling_prob:
            return vid_len
        min_len = min(max(self.min_prefix_len, 1), vid_len)
        relevant = local_segments[local_segments[:, 0] >= 0]
        if relevant.numel() == 0:
            return random.randint(min_len, vid_len)

        # Select one query as the regime anchor; all other labels stay unchanged.
        anchor = relevant[random.randrange(len(relevant))]
        start, end = (float(v) for v in anchor)
        ranges = {
            "pre": (min_len, min(vid_len, int(np.ceil(start)))),
            "active": (max(min_len, int(np.ceil(start)) + 1), min(vid_len, int(np.ceil(end)))),
            "post": (max(min_len, int(np.ceil(end)) + 1), vid_len),
        }
        feasible = [(mode, bounds, weight) for (mode, bounds), weight in zip(ranges.items(), self.prefix_state_probs) if bounds[0] <= bounds[1]]
        if not feasible:
            return random.randint(min_len, vid_len)
        bounds = random.choices(feasible, [x[2] for x in feasible], k=1)[0][1]
        return random.randint(*bounds)

    def _load_vid_feats(self, vid_id):
        if vid_id in self._video_cache:
            cached = self._video_cache.pop(vid_id)
            self._video_cache[vid_id] = cached
            return cached
        if self.vid_feat_dir.endswith('.h5'):
            if self.vid_h5 is None:
                self.vid_h5 = h5py.File(self.vid_feat_dir)
            vid_feats = self.vid_h5[vid_id][:].astype(np.float32)
        else:
            vid_feat_file = os.path.join(self.vid_feat_dir, vid_id + '.npy')
            vid_feats = np.load(vid_feat_file).astype(np.float32)

        if self.downsample_rate > 1:
            vid_feats = vid_feats[::self.downsample_rate]

        vid_feats = vid_feats.transpose()                   # (c, t)
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats))

        # normalize features to unit length
        if self.normalize_vid:
            vid_feats = F.normalize(vid_feats, dim=0)
        if self.video_cache_size > 0:
            self._video_cache[vid_id] = vid_feats
            while len(self._video_cache) > self.video_cache_size:
                self._video_cache.popitem(last=False)
        return vid_feats

    def _truncate_vid_feats(
        self,
        feats,
        segments,
        offset,
        num_trials=5000
    ):
        vid_len = feats.size(1)
        max_vid_len = self.long_term_window_size

        if vid_len <= max_vid_len:
            if self.crop_ratio is None:
                return feats, 0, vid_len

            max_vid_len = random.randint(
                int(max(np.ceil(self.crop_ratio[0] * vid_len), 1)),
                int(min(np.ceil(self.crop_ratio[1] * vid_len), vid_len))
            )
            if max_vid_len == vid_len:
                return feats, 0, vid_len

        # rough estimate on the range of valid chunks
        s0 = max(0, np.floor(segments[:, 0].min() - max_vid_len))
        s1 = min(vid_len - max_vid_len, np.ceil(segments[:, 1].max()))
        
        seg_lens = torch.clamp(segments[:, 1] - segments[:, 0], min=1e-5)

        for _ in range(num_trials):
            ws = random.randint(int(s0), int(s1)) # window start
            we = ws + max_vid_len       # window end

            # check overlap with segments
            start = torch.clamp(segments[:, 0], min=ws - offset)
            end = torch.clamp(segments[:, 1], max=we + offset)
            overlap = torch.clamp(end - start, min=0)
            if torch.all(overlap / seg_lens > self.trunc_thresh):
                feats = feats[:, ws:we]
                return feats, ws, we

        feats = feats[:, ws:we]
        return feats, ws, we

    def _load_text_feats(self, text_id):
        if self.tokenizer is not None:
            text_feats = self.tokenizer(self.text_dict[text_id]['text'])
        elif self.text_feat_dir.endswith('.h5'):
            if self.text_h5 is None:
                self.text_h5 = h5py.File(self.text_feat_dir)
            text_feats = self.text_h5[text_id][:].astype(np.float32)
            text_feats = text_feats.transpose()     # (c, t)
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
        else:
            text_feat_file = os.path.join(self.text_feat_dir, text_id + '.npy')
            text_feats = np.load(text_feat_file).astype(np.float32)
            text_feats = text_feats.transpose()     # (c, t)
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
            
        if self.is_training:
            text_feats = text_feats[:, :self.max_text_len]

        # normalize text features to unit length
        if self.normalize_text:
            text_feats = F.normalize(text_feats, dim=0)

        return text_feats

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def _build_train_samples(self):
        samples = []
        for _ in range(self.num_epochs):
            for vid_id in self.vid_dict.keys():
                video_samples = self._group(vid_id)
                shuffle(video_samples)
                if self.drop_text:
                    samples.append((vid_id, video_samples[:1]))
                    continue
                for i in range(0, len(video_samples), self.max_num_window):
                    samples.append((vid_id, video_samples[i:i+self.max_num_window]))
                if len(video_samples) % self.max_num_window != 0 and len(video_samples) > self.max_num_window:
                    samples.append((vid_id, video_samples[-self.max_num_window:]))
        return tuple(samples)

    def _build_eval_samples(self):
        samples = []
        for vid_id, vid_dict in self.vid_dict.items():
            samples += [(vid_id, [tuple(range(len(vid_dict['segments'])))])]
        return tuple(samples)

    def _group(self, vid_id):
        vid_dict = self.vid_dict[vid_id]

        if vid_dict['num_clips'] <= self.long_term_window_size:
            win_len = vid_dict['num_clips']
            if self.crop_ratio is not None:
                win_len = max(np.ceil(self.crop_ratio[0] * win_len), 1)
        else:
            win_len = self.long_term_window_size
        win_len = (
            self.clip_stride * (win_len - 1) + self.clip_size
        ) / vid_dict['fps'] 

        sort_idx = np.argsort(vid_dict['segments'][:, 0])
        segments = vid_dict['segments'][sort_idx]
        mask = np.ones(len(segments), dtype=bool)

        samples = []
        while mask.sum() > 0:
            ptr = np.nonzero(mask)[0].min()

            ws, we = segments[ptr, 0], segments[ptr, 0] + win_len
            if segments[ptr, 1] - segments[ptr, 0] > win_len:
                idx = np.array([ptr])
            else:
                is_inside = (
                    (segments[:, 0] >= ws) & (segments[:, 1] <= we) & mask
                )
                idx = np.nonzero(is_inside)[0]
                if len(idx) > self.max_num_text:
                    idx = np.random.choice(idx, self.max_num_text, replace=False)
            sample = tuple(sort_idx[idx])
            samples += [sample]
            mask[idx] = 0
        return samples

    def __len__(self):
        return len(self.data_list) // self.num_epochs

    def __getitem__(self, idx):
        vid_id, seg_idx = self.data_list[self.epoch * len(self) + idx]
        vid_dict = self.vid_dict[vid_id]

        vid_feats = self._load_vid_feats(vid_id)

        clip_size, clip_stride = self.clip_size, self.clip_stride
        clip_offset = 0.5 * clip_size / clip_stride

        segments = []
        for sid in seg_idx:
            segs = np.clip(vid_dict['segments'][np.array(sid)] * vid_dict['fps'], a_min=0, a_max=vid_dict['num_frames']) / clip_stride - clip_offset
            segs = torch.from_numpy(
                np.ascontiguousarray(segs.astype(np.float32))
            )
            rel = torch.as_tensor(vid_dict['relevance'][np.array(sid)])
            segs[~rel] = -1.0
            segments.append(segs)
        segments_cat = torch.cat(segments, dim=0)

        all_window_feats = tuple()
        all_segments = tuple()
        for segs in segments:
            if self.is_training:
                window_feats, ws, we = self._truncate_vid_feats(
                    vid_feats, segs, clip_offset
                )
                local_segments = segments_cat - ws
                prefix_len = self._sample_prefix_len(window_feats.size(1), local_segments)
                window_feats = window_feats[:, :prefix_len]
                all_window_feats += (window_feats, )
                all_segments += (local_segments, )
            else:
                all_segments = (segments_cat, )
                all_window_feats = (vid_feats, )
        all_segments = torch.stack(all_segments, dim=1)

        text_feats_list = tuple()
        relevance = tuple()
        query_ids = tuple()
        for sid in seg_idx:
            for idx in sid:
                source_text_id = vid_dict['text_ids'][idx]
                base_relevant = bool(vid_dict['relevance'][idx])
                if base_relevant:
                    query_id, augmented_relevant = self._sample_query_id(source_text_id)
                else:
                    query_id, augmented_relevant = source_text_id, False
                text_feats = self._load_text_feats(query_id)
                text_feats_list += (text_feats, )
                relevance += (base_relevant and augmented_relevant, )
                query_ids += (query_id, )


        return {
            'fps'        : vid_dict['fps'],
            'num_frames' : vid_dict['num_frames'],
            'duration'   : vid_dict['duration'],
            'segment'    : vid_dict['segments'],
            'clip_size'  : clip_size,
            'clip_stride': clip_stride,
            'target'     : all_segments,
            'vid'        : all_window_feats,
            'text'       : text_feats_list,
            'is_relevant': torch.as_tensor(relevance, dtype=torch.bool),
            'query_ids'  : query_ids,
            'official_query_ids': tuple(self.text_dict[q]['official_query_id'] for q in query_ids),
            'is_strong': tuple(self.text_dict[q]['is_strong'] for q in query_ids),
            'video_id': vid_id,
        }

class GloVeTokenizer:

    def __init__(self, name='6B'):
        import torchtext
        from torchtext.data import get_tokenizer

        cache = os.environ.get(
            'SURVVTG_GLOVE_CACHE',
            os.path.expanduser('~/.cache/survvtg/glove')
        )
        self.vocab = torchtext.vocab.GloVe(name=name, cache=cache)
        self.tokenizer = get_tokenizer("basic_english")

    def __call__(self, text, max_len=None):
        words = self.tokenizer(text)
        feats = self.vocab.get_vecs_by_tokens(words, lower_case_backup=True)
        if max_len is not None:
            feats = feats[:max_len]
        feats = feats.transpose(0, 1)

        return feats

def make_dataset(opt, num_epochs=1, is_training=True):
    opt = deepcopy(opt)
    return OnVTGDataset(is_training=is_training, num_epochs=num_epochs,  **opt)

def trivial_batch_collator(batch):
    return batch


def worker_init_reset_seed(worker_id, num_workers, rank):
    seed = torch.initial_seed() % 2 ** 31
    worker_seed = num_workers * rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)

    
def make_dataloader(
    dataset,
    generator,
    batch_size,
    num_workers,
    is_training,
    world_size=1,
    rank=0,
):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=is_training, drop_last=is_training)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=partial(worker_init_reset_seed, num_workers, rank),
        sampler=sampler,
        shuffle=(sampler is None and is_training),
        drop_last=is_training,
        generator=generator,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader, sampler
