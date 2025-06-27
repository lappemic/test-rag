# Max-Marginal Relevance (MMR) Feature

## Overview

The Swiss Legal Chatbot now includes **Max-Marginal Relevance (MMR)** re-ranking to improve the diversity of retrieved legal documents and avoid near-duplicate chunks. This helps surface varied perspectives and reduces redundant information in responses.

## What is MMR?

MMR is an algorithm that balances **relevance** and **diversity** when selecting documents from a larger candidate set. Instead of just taking the top-K most similar documents, MMR iteratively selects documents that:

1. Are relevant to the user's query
2. Are different from already selected documents

**Formula**: `MMR(d) = λ * Sim(d, query) - (1-λ) * max(Sim(d, d_i))` for `d_i` in selected documents

## Configuration

The MMR feature is controlled by several settings in `config/settings.py`:

```python
# Max-Marginal Relevance (MMR) settings
ENABLE_MMR = True                        # Enable/disable MMR re-ranking
MMR_LAMBDA = 0.5                        # Balance relevance vs diversity (0.0-1.0)
MMR_FETCH_K = 15                        # Documents to fetch before re-ranking
MMR_USE_FAST_MODE = True                 # Use heuristic vs embedding-based MMR
```

### Parameters Explained

- **`ENABLE_MMR`**: Turn MMR on/off globally
- **`MMR_LAMBDA`**: Controls the relevance vs diversity trade-off:
  - `0.0` = Maximum diversity (least redundant results)
  - `1.0` = Maximum relevance (most similar to query)
  - `0.5` = Balanced approach (recommended default)
- **`MMR_FETCH_K`**: Number of candidate documents to retrieve before MMR re-ranking
  - Should be larger than `MAX_RESULTS` (default: 5)
  - Larger values give MMR more candidates to choose from
- **`MMR_USE_FAST_MODE`**: Choose between two MMR implementations:
  - `True` = Fast heuristic-based MMR (recommended for production)
  - `False` = Full embedding-based MMR (more accurate but slower)

## How It Works

1. **Initial Retrieval**: Fetch `MMR_FETCH_K` most similar documents (e.g., 15)
2. **MMR Re-ranking**: Apply MMR algorithm to select `MAX_RESULTS` diverse documents (e.g., 5)
3. **Response Generation**: Use the diverse set of documents for answer generation

## Benefits for Legal Texts

Legal documents often contain:
- Repetitive language and similar phrasing
- Multiple articles covering related topics
- Near-duplicate content across different laws

MMR helps by:
- ✅ Reducing redundant information in responses
- ✅ Surfacing different legal perspectives
- ✅ Including relevant content from multiple laws
- ✅ Improving answer comprehensiveness

## Performance Considerations

### Fast Mode (Default)
- **Speed**: Minimal overhead (~5-10% slower than traditional ranking)
- **Method**: Uses heuristic-based similarity approximation
- **Use case**: Production environments, real-time queries

### Full Mode
- **Speed**: Moderate overhead (~20-50% slower)
- **Method**: Computes actual document embeddings for similarity
- **Use case**: When highest quality diversity is needed

## Testing

Run the MMR test suite to evaluate performance:

```bash
python scripts/test_mmr.py
```

This will:
- Compare MMR vs traditional ranking
- Test different λ parameter values
- Show diversity improvements
- Measure performance impact

## Example Results

**Traditional Ranking** (top 5 most similar):
1. Asylgesetz (AsylG) - Art. 3
2. Asylgesetz (AsylG) - Art. 5
3. Asylgesetz (AsylG) - Art. 7
4. Asylgesetz (AsylG) - Art. 9
5. Asylgesetz (AsylG) - Art. 12

**MMR Ranking** (diverse selection):
1. Asylgesetz (AsylG) - Art. 3
2. Ausländer- und Integrationsgesetz (AIG) - Art. 15
3. Bundesgesetz über das Bürgerrecht (BüG) - Art. 8
4. Schengen-Assoziierungsabkommen - Art. 4
5. Asylgesetz (AsylG) - Art. 9

## Tuning Guidelines

### For More Diversity
- Decrease `MMR_LAMBDA` (e.g., 0.3)
- Increase `MMR_FETCH_K` (e.g., 20)

### For More Relevance
- Increase `MMR_LAMBDA` (e.g., 0.7)
- Keep `MMR_FETCH_K` moderate (e.g., 12)

### For Better Performance
- Use `MMR_USE_FAST_MODE = True`
- Keep `MMR_FETCH_K` reasonable (e.g., 15)

## Integration

MMR is automatically applied in both:
- Regular queries (`query_rag`)
- Streaming queries (`query_rag_streaming`)

No changes needed to existing application code.

## Monitoring

The system logs MMR operations:

```
INFO - Applying MMR re-ranking to 15 results (λ=0.5)
INFO - MMR re-ranking completed: selected 5 diverse results
```

## Troubleshooting

### MMR Not Working
- Check `ENABLE_MMR = True` in settings
- Ensure `MMR_FETCH_K > MAX_RESULTS`
- Verify `scikit-learn` and `numpy` are installed

### Poor Diversity
- Lower `MMR_LAMBDA` value
- Increase `MMR_FETCH_K` for more candidates
- Try `MMR_USE_FAST_MODE = False` for better similarity calculation

### Performance Issues
- Use `MMR_USE_FAST_MODE = True`
- Reduce `MMR_FETCH_K` value
- Consider disabling for very time-sensitive queries

## Dependencies

The MMR feature requires:
- `scikit-learn` (for cosine similarity)
- `numpy` (for array operations)

These are automatically included in `requirements.txt`. 