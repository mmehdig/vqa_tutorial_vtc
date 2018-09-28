import logging
import os
import sys

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray

logger = logging.getLogger(__name__)

if sys.version_info[0] >= 3:
    unicode = str

xrange = range

def make_closing(base, **attrs):
    if not hasattr(base, '__enter__'):
        attrs['__enter__'] = lambda self: self
    if not hasattr(base, '__exit__'):
        attrs['__exit__'] = lambda self, type, value, traceback: self.close()
    return type('Closing' + base.__name__, (base, object), attrs)

def smart_open(fname, mode='rb'):
    _, ext = os.path.splitext(fname)
    if ext == '.bz2':
        from bz2 import BZ2File
        return make_closing(BZ2File)(fname, mode)
    if ext == '.gz':
        from gzip import GzipFile
        return make_closing(GzipFile)(fname, mode)
    return open(fname, mode)



def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')
to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)
to_unicode = any2unicode


def call_on_class_only(*args, **kwargs):
    """Raise exception when load methods are called on instance"""
    raise AttributeError('This method should be called on a class object.')


class SaveLoad(object):
    """
    Objects which inherit from this class have save/load functions, which un/pickle
    them to disk.

    This uses pickle for de/serializing, so objects must not contain
    unpicklable attributes, such as lambda functions etc.

    """
    @classmethod
    def load(cls, fname, mmap=None):
        """
        Load a previously saved object from file (also see `save`).

        If the object was saved with large arrays stored separately, you can load
        these arrays via mmap (shared memory) using `mmap='r'`. Default: don't use
        mmap, load large arrays as normal objects.

        If the file being loaded is compressed (either '.gz' or '.bz2'), then
        `mmap=None` must be set.  Load will raise an `IOError` if this condition
        is encountered.

        """
        logger.info("loading %s object from %s" % (cls.__name__, fname))

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        obj = unpickle(fname)
        obj._load_specials(fname, mmap, compress, subname)
        logger.info("loaded %s", fname)
        return obj

    def _load_specials(self, fname, mmap, compress, subname):
        """
        Loads any attributes that were stored specially, and gives the same
        opportunity to recursively included SaveLoad instances.

        """
        mmap_error = lambda x, y: IOError(
            'Cannot mmap compressed object %s in file %s. ' % (x, y) +
            'Use `load(fname, mmap=None)` or uncompress files manually.')

        for attrib in getattr(self, '__recursive_saveloads', []):
            cfname = '.'.join((fname, attrib))
            logger.info("loading %s recursively from %s.* with mmap=%s" % (
                attrib, cfname, mmap))
            getattr(self, attrib)._load_specials(cfname, mmap, compress, subname)

        for attrib in getattr(self, '__numpys', []):
            logger.info("loading %s from %s with mmap=%s" % (
                attrib, subname(fname, attrib), mmap))

            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                val = np.load(subname(fname, attrib))['val']
            else:
                val = np.load(subname(fname, attrib), mmap_mode=mmap)

            setattr(self, attrib, val)

        for attrib in getattr(self, '__scipys', []):
            logger.info("loading %s from %s with mmap=%s" % (
                attrib, subname(fname, attrib), mmap))
            sparse = unpickle(subname(fname, attrib))
            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                with np.load(subname(fname, attrib, 'sparse')) as f:
                    sparse.data = f['data']
                    sparse.indptr = f['indptr']
                    sparse.indices = f['indices']
            else:
                sparse.data = np.load(subname(fname, attrib, 'data'), mmap_mode=mmap)
                sparse.indptr = np.load(subname(fname, attrib, 'indptr'), mmap_mode=mmap)
                sparse.indices = np.load(subname(fname, attrib, 'indices'), mmap_mode=mmap)

            setattr(self, attrib, sparse)

        for attrib in getattr(self, '__ignoreds', []):
            logger.info("setting ignored attribute %s to None" % (attrib))
            setattr(self, attrib, None)

    @staticmethod
    def _adapt_by_suffix(fname):
        """Give appropriate compress setting and filename formula"""
        if fname.endswith('.gz') or fname.endswith('.bz2'):
            compress = True
            subname = lambda *args: '.'.join(list(args) + ['npz'])
        else:
            compress = False
            subname = lambda *args: '.'.join(list(args) + ['npy'])
        return (compress, subname)

    def _smart_save(self, fname, separately=None, sep_limit=10 * 1024**2,
                    ignore=frozenset(), pickle_protocol=2):
        """
        Save the object to file (also see `load`).

        If `separately` is None, automatically detect large
        numpy/scipy.sparse arrays in the object being stored, and store
        them into separate files. This avoids pickle memory errors and
        allows mmap'ing large arrays back on load efficiently.

        You can also set `separately` manually, in which case it must be
        a list of attribute names to be stored in separate files. The
        automatic check is not performed in this case.

        `ignore` is a set of attribute names to *not* serialize (file
        handles, caches etc). On subsequent load() these attributes will
        be set to None.

        `pickle_protocol` defaults to 2 so the pickled object can be imported
        in both Python 2 and 3.

        """
        logger.info(
            "saving %s object under %s, separately %s" % (
                self.__class__.__name__, fname, separately))

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        restores = self._save_specials(fname, separately, sep_limit, ignore, pickle_protocol,
                                       compress, subname)
        try:
            pickle(self, fname, protocol=pickle_protocol)
        finally:
            # restore attribs handled specially
            for obj, asides in restores:
                for attrib, val in iteritems(asides):
                    setattr(obj, attrib, val)
        logger.info("saved %s", fname)

    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        """
        Save aside any attributes that need to be handled separately, including
        by recursion any attributes that are themselves SaveLoad instances.

        Returns a list of (obj, {attrib: value, ...}) settings that the caller
        should use to restore each object's attributes that were set aside
        during the default pickle().

        """
        asides = {}
        sparse_matrices = (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)
        if separately is None:
            separately = []
            for attrib, val in iteritems(self.__dict__):
                if isinstance(val, np.ndarray) and val.size >= sep_limit:
                    separately.append(attrib)
                elif isinstance(val, sparse_matrices) and val.nnz >= sep_limit:
                    separately.append(attrib)

        # whatever's in `separately` or `ignore` at this point won't get pickled
        for attrib in separately + list(ignore):
            if hasattr(self, attrib):
                asides[attrib] = getattr(self, attrib)
                delattr(self, attrib)

        recursive_saveloads = []
        restores = []
        for attrib, val in iteritems(self.__dict__):
            if hasattr(val, '_save_specials'):  # better than 'isinstance(val, SaveLoad)' if IPython reloading
                recursive_saveloads.append(attrib)
                cfname = '.'.join((fname, attrib))
                restores.extend(val._save_specials(
                    cfname, None, sep_limit, ignore,
                    pickle_protocol, compress, subname))

        try:
            numpys, scipys, ignoreds = [], [], []
            for attrib, val in iteritems(asides):
                if isinstance(val, np.ndarray) and attrib not in ignore:
                    numpys.append(attrib)
                    logger.info("storing np array '%s' to %s" % (
                        attrib, subname(fname, attrib)))

                    if compress:
                        np.savez_compressed(subname(fname, attrib), val=np.ascontiguousarray(val))
                    else:
                        np.save(subname(fname, attrib), np.ascontiguousarray(val))

                elif isinstance(val, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)) and attrib not in ignore:
                    scipys.append(attrib)
                    logger.info("storing scipy.sparse array '%s' under %s" % (
                        attrib, subname(fname, attrib)))

                    if compress:
                        np.savez_compressed(
                            subname(fname, attrib, 'sparse'),
                            data=val.data,
                            indptr=val.indptr,
                            indices=val.indices)
                    else:
                        np.save(subname(fname, attrib, 'data'), val.data)
                        np.save(subname(fname, attrib, 'indptr'), val.indptr)
                        np.save(subname(fname, attrib, 'indices'), val.indices)

                    data, indptr, indices = val.data, val.indptr, val.indices
                    val.data, val.indptr, val.indices = None, None, None

                    try:
                        # store array-less object
                        pickle(val, subname(fname, attrib), protocol=pickle_protocol)
                    finally:
                        val.data, val.indptr, val.indices = data, indptr, indices
                else:
                    logger.info("not storing attribute %s" % (attrib))
                    ignoreds.append(attrib)

            self.__dict__['__numpys'] = numpys
            self.__dict__['__scipys'] = scipys
            self.__dict__['__ignoreds'] = ignoreds
            self.__dict__['__recursive_saveloads'] = recursive_saveloads
        except:
            # restore the attributes if exception-interrupted
            for attrib, val in iteritems(asides):
                setattr(self, attrib, val)
            raise
        return restores + [(self, asides)]

    def save(self, fname_or_handle, separately=None, sep_limit=10 * 1024**2,
             ignore=frozenset(), pickle_protocol=2):
        """
        Save the object to file (also see `load`).

        `fname_or_handle` is either a string specifying the file name to
        save to, or an open file-like object which can be written to. If
        the object is a file handle, no special array handling will be
        performed; all attributes will be saved to the same file.

        If `separately` is None, automatically detect large
        numpy/scipy.sparse arrays in the object being stored, and store
        them into separate files. This avoids pickle memory errors and
        allows mmap'ing large arrays back on load efficiently.

        You can also set `separately` manually, in which case it must be
        a list of attribute names to be stored in separate files. The
        automatic check is not performed in this case.

        `ignore` is a set of attribute names to *not* serialize (file
        handles, caches etc). On subsequent load() these attributes will
        be set to None.

        `pickle_protocol` defaults to 2 so the pickled object can be imported
        in both Python 2 and 3.

        """
        try:
            _pickle.dump(self, fname_or_handle, protocol=pickle_protocol)
            logger.info("saved %s object" % self.__class__.__name__)
        except TypeError:  # `fname_or_handle` does not have write attribute
            self._smart_save(fname_or_handle, separately, sep_limit, ignore,
                             pickle_protocol=pickle_protocol)
#endclass SaveLoad


class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class KeyedVectors(SaveLoad):
    """
    Class to contain vectors and vocab for the Word2Vec training class and other w2v methods not directly
    involved in training such as most_similar()
    """
    def __init__(self):
        self.syn0 = []
        self.syn0norm = None
        self.vocab = {}
        self.index2word = []
        self.vector_size = None

    @property
    def wv(self):
        return self

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])
        super(KeyedVectors, self).save(*args, **kwargs)

    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

         `fname` is the file used to save the vectors in
         `fvocab` is an optional file used to save the vocabulary
         `binary` is an optional boolean indicating whether the data is to be saved
         in binary word2vec format (default: False)
         `total_vec` is an optional parameter to explicitly specify total no. of vectors
         (in case word vectors are appended with document vectors afterwards)

        """
        if total_vec is None:
            total_vec = len(self.vocab)
        vector_size = self.syn0.shape[1]
        if fvocab is not None:
            logger.info("storing vocabulary in %s" % (fvocab))
            with smart_open(fvocab, 'wb') as vout:
                for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                    vout.write(to_utf8("%s %s\n" % (word, vocab.count)))
        logger.info("storing %sx%s projection weights into %s" % (total_vec, vector_size, fname))
        assert (len(self.vocab), vector_size) == self.syn0.shape
        with smart_open(fname, 'wb') as fout:
            fout.write(to_utf8("%s %s\n" % (total_vec, vector_size)))
            # store in sorted order: most frequent words at the top
            for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                row = self.syn0[vocab.index]
                if binary:
                    fout.write(to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))


    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                             limit=None, datatype=REAL):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        `norm_only` is a boolean indicating whether to only store normalised word2vec vectors in memory.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).

        If you trained the C model using non-utf8 encoding for words, specify that
        encoding in `encoding`.

        `unicode_errors`, default 'strict', is a string suitable to be passed as the `errors`
        argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source
        file may include word tokens truncated in the middle of a multibyte unicode character
        (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.

        `limit` sets a maximum number of word-vectors to read from the file. The default,
        None, means read all.

        `datatype` (experimental) can coerce dimensions to a non-default float type (such
        as np.float16) to save memory. (Such types may result in much slower bulk operations
        or incompatibility with optimized routines.)

        """
        counts = None
        if fvocab is not None:
            logger.info("loading word counts from %s", fvocab)
            counts = {}
            with smart_open(fvocab) as fin:
                for line in fin:
                    word, count = to_unicode(line).strip().split()
                    counts[word] = int(count)

        logger.info("loading projection weights from %s", fname)
        with smart_open(fname) as fin:
            header = to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
            if limit:
                vocab_size = min(vocab_size, limit)
            result = cls()
            result.vector_size = vector_size
            result.syn0 = zeros((vocab_size, vector_size), dtype=datatype)

            def add_word(word, weights):
                word_id = len(result.vocab)
                if word in result.vocab:
                    logger.warning("duplicate word '%s' in %s, ignoring all but first", word, fname)
                    return
                if counts is None:
                    # most common scenario: no vocab file given. just make up some bogus counts, in descending order
                    result.vocab[word] = Vocab(index=word_id, count=vocab_size - word_id)
                elif word in counts:
                    # use count from the vocab file
                    result.vocab[word] = Vocab(index=word_id, count=counts[word])
                else:
                    # vocab file given, but word is missing -- set count to None (TODO: or raise?)
                    logger.warning("vocabulary file is incomplete: '%s' is missing", word)
                    result.vocab[word] = Vocab(index=word_id, count=None)
                result.syn0[word_id] = weights
                result.index2word.append(word)

            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                for line_no in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch == b'':
                            raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                            word.append(ch)
                    word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = fromstring(fin.read(binary_len), dtype=REAL)
                    add_word(word, weights)
            else:
                for line_no in xrange(vocab_size):
                    line = fin.readline()
                    if line == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    parts = to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                    word, weights = parts[0], list(map(REAL, parts[1:]))
                    add_word(word, weights)
        if result.syn0.shape[0] != len(result.vocab):
            logger.info(
                "duplicate words detected, shrinking matrix size from %i to %i",
                result.syn0.shape[0], len(result.vocab)
            )
            result.syn0 = ascontiguousarray(result.syn0[: len(result.vocab)])
        assert (len(result.vocab), vector_size) == result.syn0.shape

        logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
        return result

    def word_vec(self, word, use_norm=False):
        """
        Accept a single word as input.
        Returns the word's representations in vector space, as a 1D numpy array.

        If `use_norm` is True, returns the normalized word vector.

        Example::

          >>> trained_model['office']
          array([ -1.40128313e-02, ...])

        """
        if word in self.vocab:
            if use_norm:
                return self.syn0norm[self.vocab[word].index]
            else:
                return self.syn0[self.vocab[word].index]
        else:
            raise KeyError("word '%s' not in vocabulary" % word)

    def most_similar(self, positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        If topn is False, most_similar returns the vector of similarity scores.

        `restrict_vocab` is an optional integer which limits the range of vectors which
        are searched for most-similar values. For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order. (This may be
        meaningful if you've sorted the vocabulary by descending frequency.)

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative
        ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            else:
                mean.append(weight * self.word_vec(word, use_norm=True))
                if word in self.vocab:
                    all_words.add(self.vocab[word].index)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        limited = self.syn0norm if restrict_vocab is None else self.syn0norm[:restrict_vocab]
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def wmdistance(self, document1, document2):
        """
        Compute the Word Mover's Distance between two documents. When using this
        code, please consider citing the following papers:

        .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
        .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
        .. Matt Kusner et al. "From Word Embeddings To Document Distances".

        Note that if one of the documents have no words that exist in the
        Word2Vec vocab, `float('inf')` (i.e. infinity) will be returned.

        This method only works if `pyemd` is installed (can be installed via pip, but requires a C compiler).

        Example:
            >>> # Train word2vec model.
            >>> model = Word2Vec(sentences)

            >>> # Some sentences to test.
            >>> sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
            >>> sentence_president = 'The president greets the press in Chicago'.lower().split()

            >>> # Remove their stopwords.
            >>> from nltk.corpus import stopwords
            >>> stopwords = nltk.corpus.stopwords.words('english')
            >>> sentence_obama = [w for w in sentence_obama if w not in stopwords]
            >>> sentence_president = [w for w in sentence_president if w not in stopwords]

            >>> # Compute WMD.
            >>> distance = model.wmdistance(sentence_obama, sentence_president)
        """

        if not PYEMD_EXT:
            raise ImportError("Please install pyemd Python package to compute WMD.")

        # Remove out-of-vocabulary words.
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in self]
        document2 = [token for token in document2 if token in self]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).',
                        diff1, diff2)

        if len(document1) == 0 or len(document2) == 0:
            logger.info('At least one of the documents had no words that were'
                        'in the vocabulary. Aborting (returning inf).')
            return float('inf')

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed by a single unique token
            return 0.0

        # Sets for faster look-up.
        docset1 = set(document1)
        docset2 = set(document2)

        # Compute distance matrix.
        distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if not t1 in docset1 or not t2 in docset2:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = sqrt(np_sum((self[t1] - self[t2])**2))

        if np_sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def nbow(document):
            d = zeros(vocab_len, dtype=double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents.
        d1 = nbow(document1)
        d2 = nbow(document2)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)

    def most_similar_cosmul(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words, using the multiplicative combination objective
        proposed by Omer Levy and Yoav Goldberg in [4]_. Positive words still contribute
        positively towards the similarity, negative words negatively, but with less
        susceptibility to one large distance dominating the calculation.

        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

        Additional positive or negative examples contribute to the numerator or denominator,
        respectively – a potentially sensible but untested extension of the method. (With
        a single positive example, rankings will be the same as in the default most_similar.)

        Example::

          >>> trained_model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'])
          [(u'iraq', 0.8488819003105164), ...]

        .. [4] Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
            positive = [positive]

        all_words = set([self.vocab[word].index for word in positive+negative
            if not isinstance(word, ndarray) and word in self.vocab])

        positive = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in positive
        ]
        negative = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in negative
        ]

        if not positive:
            raise ValueError("cannot compute similarity with no input")

        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + dot(self.syn0norm, term)) / 2) for term in positive]
        neg_dists = [((1 + dot(self.syn0norm, term)) / 2) for term in negative]
        dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words.

        If topn is False, similar_by_word returns the vector of similarity scores.

        `restrict_vocab` is an optional integer which limits the range of vectors which
        are searched for most-similar values. For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order. (This may be
        meaningful if you've sorted the vocabulary by descending frequency.)

        Example::

          >>> trained_model.similar_by_word('graph')
          [('user', 0.9999163150787354), ...]

        """

        return self.most_similar(positive=[word], topn=topn, restrict_vocab=restrict_vocab)

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words by vector.

        If topn is False, similar_by_vector returns the vector of similarity scores.

        `restrict_vocab` is an optional integer which limits the range of vectors which
        are searched for most-similar values. For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order. (This may be
        meaningful if you've sorted the vocabulary by descending frequency.)

        Example::

          >>> trained_model.similar_by_vector([1,2])
          [('survey', 0.9942699074745178), ...]

        """

        return self.most_similar(positive=[vector], topn=topn, restrict_vocab=restrict_vocab)

    def doesnt_match(self, words):
        """
        Which word from the given list doesn't go with the others?

        Example::

          >>> trained_model.doesnt_match("breakfast cereal dinner lunch".split())
          'cereal'

        """
        self.init_sims()

        used_words = [word for word in words if word in self]
        if len(used_words) != len(words):
            ignored_words = set(words) - set(used_words)
            logger.warning("vectors for words %s are not present in the model, ignoring these words", ignored_words)
        if not used_words:
            raise ValueError("cannot select a word from an empty list")
        vectors = vstack(self.word_vec(word, use_norm=True) for word in used_words).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, used_words))[0][1]

    def __getitem__(self, words):

        """
        Accept a single word or a list of words as input.

        If a single word: returns the word's representations in vector space, as
        a 1D numpy array.

        Multiple words: return the words' representations in vector space, as a
        2d numpy array: #words x #vector_size. Matrix rows are in the same order
        as in input.

        Example::

          >>> trained_model['office']
          array([ -1.40128313e-02, ...])

          >>> trained_model[['office', 'products']]
          array([ -1.40128313e-02, ...]
                [ -1.70425311e-03, ...]
                 ...)

        """
        if isinstance(words, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.word_vec(words)

        return vstack([self.word_vec(word) for word in words])

    def __contains__(self, word):
        return word in self.vocab

    def similarity(self, w1, w2):
        """
        Compute cosine similarity between two words.

        Example::

          >>> trained_model.similarity('woman', 'man')
          0.73723527

          >>> trained_model.similarity('woman', 'woman')
          1.0

        """
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def n_similarity(self, ws1, ws2):
        """
        Compute cosine similarity between two sets of words.

        Example::

          >>> trained_model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
          0.61540466561049689

          >>> trained_model.n_similarity(['restaurant', 'japanese'], ['japanese', 'restaurant'])
          1.0000000000000004

          >>> trained_model.n_similarity(['sushi'], ['restaurant']) == trained_model.similarity('sushi', 'restaurant')
          True

        """
        if not(len(ws1) and len(ws2)):
            raise ZeroDivisionError('Atleast one of the passed list is empty.')
        v1 = [self[word] for word in ws1]
        v2 = [self[word] for word in ws2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)),
                   matutils.unitvec(array(v2).mean(axis=0)))

    @staticmethod
    def log_accuracy(section):
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            logger.info("%s: %.1f%% (%i/%i)" %
                        (section['section'], 100.0 * correct / (correct + incorrect),
                         correct, correct + incorrect))

    def accuracy(self, questions, restrict_vocab=30000, most_similar=most_similar, case_insensitive=True):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See questions-words.txt in https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word not in the first `restrict_vocab`
        words (default 30,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
        In case `case_insensitive` is True, the first `restrict_vocab` words are taken first, and then
        case normalization is performed.

        Use `case_insensitive` to convert all words in questions and vocab to their uppercase form before
        evaluating the accuracy (default True). Useful in case of case-mismatch between training tokens
        and question words. In case of multiple case variants of a single word, the vector for the first
        occurrence (also the most frequent if vocabulary is sorted) is taken.

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
        ok_vocab = dict((w.upper(), v) for w, v in reversed(ok_vocab)) if case_insensitive else dict(ok_vocab)

        sections, section = [], None
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self.log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    if case_insensitive:
                        a, b, c, expected = [word.upper() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except:
                    logger.info("skipping invalid line #%i in %s" % (line_no, questions))
                    continue
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    logger.debug("skipping line #%i with OOV words: %s" % (line_no, line.strip()))
                    continue

                original_vocab = self.vocab
                self.vocab = ok_vocab
                ignore = set([a, b, c])  # input words to be ignored
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                sims = most_similar(self, positive=[b, c], negative=[a], topn=False, restrict_vocab=restrict_vocab)
                self.vocab = original_vocab
                for index in matutils.argsort(sims, reverse=True):
                    predicted = self.index2word[index].upper() if case_insensitive else self.index2word[index]
                    if predicted in ok_vocab and predicted not in ignore:
                        if predicted != expected:
                            logger.debug("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                        break
                if predicted == expected:
                    section['correct'].append((a, b, c, expected))
                else:
                    section['incorrect'].append((a, b, c, expected))
        if section:
            # store the last section, too
            sections.append(section)
            self.log_accuracy(section)

        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
        }
        self.log_accuracy(total)
        sections.append(total)
        return sections

    @staticmethod
    def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
        logger.info('Pearson correlation coefficient against %s: %.4f', pairs, pearson[0])
        logger.info('Spearman rank-order correlation coefficient against %s: %.4f', pairs, spearman[0])
        logger.info('Pairs with unknown words ratio: %.1f%%', oov)

    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000, case_insensitive=True,
                            dummy4unknown=False):
        """
        Compute correlation of the model with human similarity judgments. `pairs` is a filename of a dataset where
        lines are 3-tuples, each consisting of a word pair and a similarity value, separated by `delimiter`.
        An example dataset is included in Gensim (test/test_data/wordsim353.tsv). More datasets can be found at
        http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html or https://www.cl.cam.ac.uk/~fh295/simlex.html.

        The model is evaluated using Pearson correlation coefficient and Spearman rank-order correlation coefficient
        between the similarities from the dataset and the similarities produced by the model itself.
        The results are printed to log and returned as a triple (pearson, spearman, ratio of pairs with unknown words).

        Use `restrict_vocab` to ignore all word pairs containing a word not in the first `restrict_vocab`
        words (default 300,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
        If `case_insensitive` is True, the first `restrict_vocab` words are taken, and then case normalization
        is performed.

        Use `case_insensitive` to convert all words in the pairs and vocab to their uppercase form before
        evaluating the model (default True). Useful when you expect case-mismatch between training tokens
        and words pairs in the dataset. If there are multiple case variants of a single word, the vector for the first
        occurrence (also the most frequent if vocabulary is sorted) is taken.

        Use `dummy4unknown=True` to produce zero-valued similarities for pairs with out-of-vocabulary words.
        Otherwise (default False), these pairs are skipped entirely.
        """
        ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
        ok_vocab = dict((w.upper(), v) for w, v in reversed(ok_vocab)) if case_insensitive else dict(ok_vocab)

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_vocab = self.vocab
        self.vocab = ok_vocab

        for line_no, line in enumerate(utils.smart_open(pairs)):
            line = utils.to_unicode(line)
            if line.startswith('#'):
                # May be a comment
                continue
            else:
                try:
                    if case_insensitive:
                        a, b, sim = [word.upper() for word in line.split(delimiter)]
                    else:
                        a, b, sim = [word for word in line.split(delimiter)]
                    sim = float(sim)
                except:
                    logger.info('skipping invalid line #%d in %s', line_no, pairs)
                    continue
                if a not in ok_vocab or b not in ok_vocab:
                    oov += 1
                    if dummy4unknown:
                        similarity_model.append(0.0)
                        similarity_gold.append(sim)
                        continue
                    else:
                        logger.debug('skipping line #%d with OOV words: %s', line_no, line.strip())
                        continue
                similarity_gold.append(sim)  # Similarity from the dataset
                similarity_model.append(self.similarity(a, b))  # Similarity from the model
        self.vocab = original_vocab
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

        logger.debug(
            'Pearson correlation coefficient against %s: %f with p-value %f',
            pairs, pearson[0], pearson[1]
        )
        logger.debug(
            'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
            pairs, spearman[0], spearman[1]
        )
        logger.debug('Pairs with unknown words: %d' % oov)
        self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
        return pearson, spearman, oov_ratio


    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.syn0.shape[0]):
                    self.syn0[i, :] /= sqrt((self.syn0[i, :] ** 2).sum(-1))
                self.syn0norm = self.syn0
            else:
                self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def get_embedding_layer(self, train_embeddings=False):
        """
        Return a Keras 'Embedding' layer with weights set as the Word2Vec model's learned word embeddings
        """
        if not KERAS_INSTALLED:
            raise ImportError("Please install Keras to use this function")
        weights = self.syn0
        layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])  # No extra mem usage here as `Embedding` layer doesn't create any new matrix for weights
        return layer
