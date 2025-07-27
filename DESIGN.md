# Drosophila Grooming Analysis Pipeline: Design Document (v2)

## 1. Project Overview

### 1.1 Problem Statement
Manual scoring of Drosophila grooming behavior in time-lapse microscopy videos is extremely time-intensive (1 hour per 9000-frame video). With insufficient camera resolution for automated neural network classification, we need an optimized manual sampling strategy that maintains statistical power while dramatically reducing analysis time.

### 1.2 Solution Design
A three-stage pipeline that:
1. Determines optimal sampling parameters from pilot data
2. Generates sampled videos for manual scoring
3. Performs comprehensive statistical analysis comparing genotypes

### 1.3 Key Constraints
* Videos: Default 9000 frames at 30 fps (5 minutes), configurable
* Each arena contains flies of identical genetic background, age, and sex
* Target: <20% false negative rate with <5% false positive rate
* Edge events must be handled consistently
* All recordings performed during morning sessions

### 1.4 Data Format Consistency
The pipeline uses a consistent **event-based format** for all manual annotations:
* Users provide CSV files with alternating start/stop frame numbers
* Same format for both pilot data and final scored data
* Single column labeled 'Frame' containing frame numbers where grooming events begin and end
* This format minimizes manual effort - users only mark transitions, not every frame

Example annotation workflow:
1. User watches video and notes: "Grooming starts at frame 234, stops at frame 567"
2. User creates CSV with these transition points
3. Scripts convert to frame-by-frame format internally for analysis

### 1.5 Integration with Existing Codebase
This pipeline builds upon and extends the existing `fly_behavior_analysis` repository (https://github.com/wilya7/fly_behavior_analysis), reusing core functionality:
* CSV processing and validation routines
* Timeline generation from event-based data
* Event list creation and manipulation
* Visualization components (raster plots, box plots)
* Error handling and logging infrastructure

New components will extend this foundation with:
* Sampling optimization algorithms
* Video processing and concatenation
* Advanced statistical analysis with FDR correction
* Edge event handling and sensitivity analysis

## 2. Design Decisions and Rationale

### 2.1 Sampling Strategy
**Decision**: Use non-overlapping windows with uniform random sampling
**Rationale**:
* Non-overlapping windows prevent double-counting events
* Uniform sampling ensures unbiased temporal coverage
* Random selection maintains statistical validity for inference

### 2.2 Window Size Selection
**Decision**: Choose window size that evenly divides total frames AND captures 80% of complete grooming events
**Rationale**:
* Even division prevents partial window complications
* 80% capture ensures most events are fully observed
* For 9000 frames: viable sizes include 100, 150, 180, 200, 225, 250, 300, 360, 450, 500, 600, 750, 900
* Algorithm adapts to different total frame counts

### 2.3 Edge Event Handling and Trade-offs

#### 2.3.1 The Edge Event Problem
Edge events (grooming bouts partially captured at window boundaries) present a fundamental challenge in sampling-based analysis:

**In Pilot Data (full video analysis):**
- Only 2 edge events maximum (start and end of video)
- Event frequency estimates are highly accurate
- Bout duration estimates are slightly biased (underestimated for edge events)

**In Sampled Data:**
- Many more edge events (2 per sampled window)
- Event frequency remains accurately estimated if all events are counted
- Duration estimates are biased for edge events (truncated observations)
- Bias increases as window size decreases relative to bout duration

#### 2.3.2 Considered Approaches

**Option 1: Overlapping Windows with Deduplication**
- *Pros*: Better complete event capture, more accurate duration estimates
- *Cons*: 
  - Complex deduplication logic prone to bugs
  - Non-independent samples complicate statistics
  - Difficult to validate FDR corrections
  - Harder for reviewers to evaluate methodology
  - Significant implementation complexity

**Option 2: Non-overlapping Windows (Selected)**
- *Pros*:
  - Clean, defensible statistics
  - Independent samples enable valid bootstrap and FDR procedures
  - Straightforward implementation and validation
  - Easy to verify correctness
  - Standard approach in literature
- *Cons*:
  - More edge events than overlapping approach
  - Duration estimates require careful handling

#### 2.3.3 Dual Metric Strategy
Based on the differential impact of edge events on frequency vs duration metrics:

**For Frequency/Count Metrics:**
- **Include ALL edge events** regardless of duration
- Maximizes accuracy for event detection
- No information loss
- Unbiased estimation of grooming frequency

**For Duration/Time Metrics:**
- **Use only complete events** (non-edge events) as primary analysis
- Provides unbiased duration estimates
- If insufficient complete events (<5), include edge events with minimum duration threshold
- Always report which events were used for each metric

#### 2.3.4 Implementation
```python
# Pseudocode for dual approach
def calculate_metrics(events):
    # Frequency metrics use ALL events
    frequency_metrics = {
        'event_count': len(events),
        'grooming_frequency': len(events) / video_duration
    }
    
    # Duration metrics use complete events only
    complete_events = [e for e in events if not e['is_edge_event']]
    if len(complete_events) >= 5:
        duration_metrics = calculate_from_events(complete_events)
        reliability = 'high'
    else:
        # Fallback: use edge events with minimum duration
        filtered_events = [e for e in events if e['duration'] >= min_duration]
        duration_metrics = calculate_from_events(filtered_events)
        reliability = 'moderate'
    
    return frequency_metrics, duration_metrics, reliability
```

#### 2.3.5 Mitigation Strategies
To maximize data quality while maintaining transparency:

1. **Optimize Window Size**: Algorithm prioritizes windows ≥1.5x maximum bout duration
2. **Dual Metric Approach**: Different event sets for frequency vs duration metrics
3. **Reliability Indicators**: Report confidence level for each metric type
4. **Transparent Reporting**: Clearly document number of events used for each metric
5. **Sensitivity Analysis**: Compare results using different edge event handling strategies

**Decision**: Use dual approach - ALL edge events for frequency, complete events only for duration
**Rationale**:
* Maximizes accuracy for frequency estimation (most robust metric)
* Eliminates duration bias from truncated observations
* Provides transparent reliability indicators
* Maintains statistical validity while using all available information

### 2.4 Bootstrap Validation
**Decision**: 10,000 bootstrap iterations per parameter combination
**Rationale**:
* Provides robust confidence intervals
* Accounts for event clustering and temporal variability
* Enables optimization across multiple criteria

### 2.5 Multiple Testing Correction
**Decision**: Apply False Discovery Rate (FDR) correction using Benjamini-Hochberg procedure
**Rationale**: When comparing genotypes across multiple grooming metrics (frequency, duration, bout structure), we perform multiple statistical tests. Without correction:
* Testing 5 metrics at α=0.05 gives a 23% chance of at least one false positive
* This inflates Type I error rate unacceptably

**Example scenario without correction**:
```
Wild-type vs Mutant comparison:
- Grooming frequency: p = 0.031 ✓
- Mean bout duration: p = 0.044 ✓
- Fragmentation index: p = 0.049 ✓

Conclusion: "Mutant differs in 3/5 metrics!" (Likely false positives)
```

**With FDR correction**:
```
Same data after FDR:
- All q-values > 0.05
- Conclusion: No significant differences (protects against false discoveries)
```

**Implementation strategy**:
* Report both raw p-values and FDR-corrected q-values
* Use hierarchical approach: primary metrics (frequency, total time) vs. exploratory metrics
* Maintain transparency about correction methods

### 2.6 Folder-Based Group Organization
**Decision**: Use folder names as group identifiers for genetic background
**Rationale**:
* Eliminates manual CSV creation and potential errors
* Enforces organized data structure
* Simplifies user interface
* Maintains clear separation between experimental groups
* Folder name directly indicates genetic background in reports

### 2.7 Configurable Frame Count
**Decision**: Make total frame count a configurable parameter (default: 9000)
**Rationale**:
* Different experimental protocols may use different video lengths
* Algorithm automatically adapts window size recommendations
* Maintains flexibility for future experiments
* All calculations scale appropriately with frame count

## 3. Script Specifications

### 3.1 Script 1: `pilot_grooming_optimizer.py`
**Purpose**: Analyze pilot data to determine optimal window size and sampling percentage

**Integration with existing code**:
- Imports `process_csv`, `generate_timeline`, `generate_event_list` from existing repository
- Extends visualization functions for optimization curves
- Reuses error handling and logging infrastructure

**Input**:
* Directory path containing CSV files with event-based annotations
* Each file contains alternating start/stop frame numbers (even number of rows)
* Format: Single column labeled 'Frame' with frame numbers where grooming starts and stops
* One file per video, any filename accepted (e.g., `fly_01.csv`, `pilot_arena_2.csv`)
* `--total-frames`: Total frames per video (default: 9000)

**Example input CSV format**:
```csv
Frame
234
567
1200
1456
2300
2890
```

Where:
* Row 1-2: First grooming event (frames 234-567)
* Row 3-4: Second grooming event (frames 1200-1456)
* Row 5-6: Third grooming event (frames 2300-2890)

```python
def main(data_directory, output_path, total_frames=9000):
    """
    Main analysis pipeline for optimization
    
    Steps:
    1. Load all pilot videos
    2. Identify all grooming events and their durations
    3. Analyze grooming bout duration distribution
    4. Determine minimum edge event duration
    5. Test window sizes that evenly divide total_frames
    6. For each window size, test sampling percentages
    7. Select optimal parameters
    """
    
    # Import from existing repository
    from fly_behavior_analysis import (
        process_csv,
        generate_timeline,
        generate_event_list,
        create_visualizations
    )
    
    # Core data structure for events
    Event = {
        'video_id': str,
        'start_frame': int,
        'end_frame': int,
        'duration_frames': int,
        'is_edge_event': bool,
        'edge_type': str  # 'start', 'end', 'both', or None
    }
```

**Key Functions**:
1. `load_pilot_data(directory)` - Load all CSV files using existing `process_csv`
2. `analyze_bout_durations(all_events)` - Generate duration distribution statistics
3. `get_viable_window_sizes(total_frames, min_size=None)` - Find all divisors of total_frames
4. `recommend_min_edge_duration(min_bout_duration, total_frames)` - Find minimum edge duration that divides total_frames
5. `calculate_window_coverage(events, window_size)` - Determine % of complete events captured
6. `estimate_edge_event_impact(events, window_size, total_frames)` - Predict edge event frequency and duration bias
7. `bootstrap_sampling(events, window_size, sampling_rate, n_iterations=10000)` - Simulate sampling outcomes
8. `optimize_parameters(all_events, target_fnr=0.20, max_fpr=0.05, total_frames=9000)` - Find optimal combination

**Output**:
* JSON file with:
   * Recommended window size
   * Recommended sampling percentage
   * Recommended minimum edge event duration
   * Grooming bout duration distribution statistics
   * Performance metrics for all tested combinations
   * Per-video validation results
   * Edge event impact analysis
   * Total frames used in analysis
* PDF report with:
   * Optimization curves and confidence intervals
   * Histogram of grooming bout durations
   * Edge event frequency predictions
   * Window size vs. complete event capture plot
   * Table of divisible edge duration options

### 3.2 Script 2: `video_sampler.py`
**Purpose**: Generate sampled video containing only selected windows with visual separators

**Input**:
* Directory containing TIF video files (folder name = genetic background)
* Window size (frames)
* Sampling percentage
* Random seed (for reproducibility)
* `--total-frames`: Expected frames per video (default: 9000)

**Expected folder structure**:
```
wild_type/          # Folder name indicates genetic background
├── fly_001.tif     # Any filename accepted
├── fly_002.tif
├── experiment_03.tif
└── ...
```

**Processing**:
```python
def main(video_directory, window_size, sampling_percentage, seed=None, total_frames=9000):
    """
    Generate sampled video for manual scoring
    
    Steps:
    1. Extract group name (genetic background) from directory name
    2. For each video file (ordered by filename):
       - Load single TIF file
       - Verify frame count (warn if not total_frames)
       - Divide into non-overlapping windows
       - Randomly select specified percentage
       - Extract selected frames
    3. Create separator frames with filename (black background, white text)
    4. Concatenate: separator + windows + separator + windows...
    5. Generate metadata files including group information
    """
    
    # Extract genetic background
    genetic_background = os.path.basename(os.path.normpath(video_directory))
    
    # Metadata structure
    SampleMetadata = {
        'source_video': str,
        'window_index': int,
        'start_frame_original': int,
        'end_frame_original': int,
        'start_frame_sampled': int,
        'end_frame_sampled': int,
        'actual_total_frames': int  # Track actual vs expected
    }
    
    # Group information structure
    GroupInfo = {
        'genetic_background': str,  # From folder name
        'n_videos': int,
        'video_list': List[str],
        'sampling_parameters': {
            'window_size': int,
            'sampling_percent': float,
            'random_seed': int,
            'separator_duration_seconds': float,  # 4 seconds
            'expected_total_frames': int,
            'actual_frame_counts': Dict[str, int]  # Per-video actual counts
        },
        'creation_date': str,
        'total_frames_original': int,
        'total_frames_sampled': int
    }
```

**Key Functions**:
1. `load_tif_video(filepath)` - Read single TIF file
2. `create_separator_frame(filename, frame_shape, duration_seconds=4, fps=30)` - Generate black frames with white text
3. `generate_window_indices(n_frames, window_size)` - Create window boundaries
4. `sample_windows(windows, percentage, seed)` - Random selection
5. `concatenate_with_separators(video_chunks, filenames)` - Merge with separators
6. `write_metadata(sample_info, output_path)` - Save reconstruction map
7. `write_group_info(group_data, output_path)` - Save group metadata

**Output** (automatically created in the input directory):
```
wild_type/
├── fly_001.tif         # Original files (unchanged)
├── fly_002.tif
├── ...
├── sampled_video.tif   # New: concatenated sampled windows with separators
├── sampling_metadata.csv   # New: frame mapping for statistical analysis
└── group_info.json     # New: group and sampling information for statistical analysis
```

**sampling_metadata.csv format**:
```csv
source_video,window_index,start_frame_original,end_frame_original,start_frame_sampled,end_frame_sampled,actual_total_frames
fly_001.tif,2,600,899,0,299,9000
fly_001.tif,5,1500,1799,420,719,9000
fly_002.tif,1,300,599,840,1139,8950
...
```

### 3.3 Script 3: `grooming_statistical_analyzer.py`
**Purpose**: Comprehensive statistical analysis and genotype comparison with edge event sensitivity analysis

**Integration with existing code**:
- Leverages existing visualization functions (raster plots, box plots)
- Extends statistical capabilities with FDR correction
- Reuses timeline and event processing functions

**Input Files**:
1. **Scored data file** (created by user after manual scoring):
   * Event-based CSV format with single column labeled 'Frame'
   * Contains alternating start/stop frame numbers (even number of rows)
   * Frame numbers refer to the sampled video (not original video)
   * Example:
```csv
Frame
120
245
580
672
```
(First event: frames 120-245, Second event: frames 580-672)

2. **Metadata file** (created by `video_sampler.py`):
   * `sampling_metadata.csv`: Maps sampled frames to original video frames

3. **Group info file** (created by `video_sampler.py`):
   * `group_info.json`: Contains genetic background and sampling parameters

**Command-Line Interface**:

**Mode 1 - Single group analysis**:
```bash
python grooming_statistical_analyzer.py \
    --scored-data ./wild_type/scored_sampled.csv \
    --metadata ./wild_type/sampling_metadata.csv \
    --group-info ./wild_type/group_info.json \
    --output-dir ./wild_type/results/ \
    --fdr-method benjamini-hochberg
```

**Mode 2 - Two group comparison**:
```bash
python grooming_statistical_analyzer.py \
    --scored-data-1 ./wild_type/scored_sampled.csv \
    --metadata-1 ./wild_type/sampling_metadata.csv \
    --group-info-1 ./wild_type/group_info.json \
    --scored-data-2 ./mutant_X/scored_sampled.csv \
    --metadata-2 ./mutant_X/sampling_metadata.csv \
    --group-info-2 ./mutant_X/group_info.json \
    --output-dir ./comparison_results/ \
    --fdr-method benjamini-hochberg \
    --edge-duration-min 15
```

**Parameters**:
* `--scored-data` / `--scored-data-1/2`: Path to manually scored CSV file with event pairs (required)
* `--metadata` / `--metadata-1/2`: Path to sampling metadata from video_sampler (required)
* `--group-info` / `--group-info-1/2`: Path to group info JSON from video_sampler (required)
* `--output-dir`: Directory for results (default: `./results/` for single, `./comparison_<group1>_vs_<group2>/` for comparison)
* `--fdr-method`: Multiple testing correction method (default: `benjamini-hochberg`, options: `benjamini-hochberg`, `bonferroni`, `none`)
* `--edge-duration-min`: Minimum frames for edge event (default: from pilot optimizer or 15)
* `--alpha`: Significance level (default: 0.05)

**Processing**:
```python
def main(args):
    """
    Analyze grooming data for one or two groups
    
    Args parsed from command line:
    - Single group: scored_data, metadata, group_info, output_dir, etc.
    - Two groups: scored_data_1/2, metadata_1/2, group_info_1/2, output_dir, etc.
    
    Key features:
    - Converts event-based scoring to frame-by-frame timeline
    - Maps sampled frame numbers back to original using metadata
    - Includes all edge events in primary analysis
    - Performs sensitivity analysis excluding edge events
    - Uses genetic background from group_info.json
    - Creates output directory if it doesn't exist
    - Handles videos with different total frame counts
    """
    
    # Import from existing repository
    from fly_behavior_analysis import (
        process_csv,
        generate_timeline,
        generate_event_list,
        calculate_file_summary,
        create_raster_plot,
        create_box_plot
    )
    
    # Determine analysis mode
    if args.scored_data:
        mode = 'single'
        # Validate single group files
        validate_file_exists(args.scored_data, "Scored data")
        validate_file_exists(args.metadata, "Metadata")
        validate_file_exists(args.group_info, "Group info")
        
        # Set default output directory
        if not args.output_dir:
            group_info = load_json(args.group_info)
            args.output_dir = f"./results_{group_info['genetic_background']}/"
            
    elif args.scored_data_1 and args.scored_data_2:
        mode = 'comparison'
        # Validate comparison files
        for suffix in ['1', '2']:
            validate_file_exists(getattr(args, f'scored_data_{suffix}'), f"Scored data {suffix}")
            validate_file_exists(getattr(args, f'metadata_{suffix}'), f"Metadata {suffix}")
            validate_file_exists(getattr(args, f'group_info_{suffix}'), f"Group info {suffix}")
        
        # Set default output directory
        if not args.output_dir:
            group1_info = load_json(args.group_info_1)
            group2_info = load_json(args.group_info_2)
            args.output_dir = f"./comparison_{group1_info['genetic_background']}_vs_{group2_info['genetic_background']}/"
    else:
        raise ValueError("Must specify either single group (--scored-data, --metadata, --group-info) " +
                       "or comparison (--scored-data-1/2, --metadata-1/2, --group-info-1/2)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
```

**Key Analysis Functions**:
```python
def map_to_original_frames(timeline_sampled, metadata_path):
    """
    Convert sampled video timeline to original video context
    
    Uses metadata to:
    - Map each sampled frame to source video and original frame
    - Handle separator frames (exclude from analysis)
    - Reconstruct timing in original video context
    - Account for videos with different total frame counts
    
    Returns timeline with original frame numbers
    """

def identify_edge_events(events, window_metadata, edge_duration_min):
    """
    Mark events that touch window boundaries
    
    Edge event types:
    - 'start': Event begins at window start
    - 'end': Event ends at window end
    - 'both': Event spans entire window
    - None: Complete event within window
    
    Assigns minimum duration to edge events
    """

def calculate_individual_metrics(events, total_frames, fps=30):
    """
    Calculate per-fly metrics using dual approach for edge events
    
    Note: total_frames comes from actual video length in metadata
    
    Returns dict with:
    Frequency metrics (using ALL events):
    - grooming_frequency: events/minute
    - bout_count: number of events
    - edge_event_count: number of edge events
    - edge_event_proportion: % of total events
    
    Duration metrics (using complete events only):
    - mean_bout_duration: seconds
    - total_grooming_time: percentage of time
    - inter_bout_interval: mean seconds between events
    - fragmentation_index: CV of bout durations
    
    Metadata:
    - duration_metric_reliability: 'high' or 'moderate'
    - n_events_for_frequency: total count including edge events
    - n_events_for_duration: count of complete events used
    """
    
    # Separate events by type
    all_events = events
    complete_events = [e for e in events if not e['is_edge_event']]
    edge_events = [e for e in events if e['is_edge_event']]
    
    # FREQUENCY METRICS - use ALL events
    grooming_frequency = len(all_events) / (total_frames / fps / 60)
    bout_count = len(all_events)
    edge_event_count = len(edge_events)
    edge_event_proportion = edge_event_count / bout_count if bout_count > 0 else 0
    
    # DURATION METRICS - use complete events only
    if len(complete_events) >= 5:
        # Sufficient complete events for reliable duration estimates
        duration_events = complete_events
        duration_reliability = 'high'
    elif len(complete_events) > 0:
        # Few complete events, but use what we have
        duration_events = complete_events
        duration_reliability = 'moderate'
    else:
        # No complete events - fallback to edge events with minimum duration
        min_duration = 15  # frames, from pilot data
        duration_events = [e for e in all_events if e['duration_frames'] >= min_duration]
        duration_reliability = 'low'
    
    # Calculate duration-based metrics
    if len(duration_events) > 0:
        durations = [e['duration_frames'] / fps for e in duration_events]
        mean_bout_duration = np.mean(durations)
        total_grooming_frames = sum([e['duration_frames'] for e in duration_events])
        total_grooming_time = (total_grooming_frames / total_frames) * 100
        
        if len(durations) > 1:
            fragmentation_index = np.std(durations) / np.mean(durations)
        else:
            fragmentation_index = np.nan
            
        # Inter-bout intervals (using complete events only for accuracy)
        if len(complete_events) > 1:
            intervals = []
            sorted_events = sorted(complete_events, key=lambda x: x['start_frame'])
            for i in range(len(sorted_events) - 1):
                interval = (sorted_events[i+1]['start_frame'] - sorted_events[i]['end_frame']) / fps
                intervals.append(interval)
            inter_bout_interval = np.mean(intervals)
        else:
            inter_bout_interval = np.nan
    else:
        mean_bout_duration = 0
        total_grooming_time = 0
        fragmentation_index = np.nan
        inter_bout_interval = np.nan
    
    return {
        # Frequency metrics (ALL events)
        'grooming_frequency': grooming_frequency,
        'bout_count': bout_count,
        'edge_event_count': edge_event_count,
        'edge_event_proportion': edge_event_proportion,
        
        # Duration metrics (complete events only)
        'mean_bout_duration': mean_bout_duration,
        'total_grooming_time': total_grooming_time,
        'fragmentation_index': fragmentation_index,
        'inter_bout_interval': inter_bout_interval,
        
        # Metadata
        'duration_metric_reliability': duration_reliability,
        'n_events_for_frequency': len(all_events),
        'n_events_for_duration': len(duration_events)
    }

def perform_sensitivity_analysis(events, total_frames, fps=30):
    """
    Analyze impact of edge events on different metrics
    
    Since frequency metrics always use ALL events, sensitivity analysis
    focuses on duration metrics and alternative edge event handling strategies
    
    Returns:
    - primary_metrics: using dual approach (all events for frequency, complete for duration)
    - alternative_strategies: dict of metrics using different approaches
    - edge_event_impact: detailed breakdown of edge effects
    """
    
    # Primary analysis (dual approach)
    primary_metrics = calculate_individual_metrics(events, total_frames, fps)
    
    # Alternative strategies for comparison
    alternative_strategies = {
        'exclude_all_edge': {
            'description': 'Exclude all edge events (old approach)',
            'metrics': calculate_metrics_excluding_edge(events, total_frames, fps)
        },
        'include_all_edge': {
            'description': 'Include all edge events for all metrics',
            'metrics': calculate_metrics_including_all(events, total_frames, fps)
        },
        'minimum_duration_threshold': {
            'description': 'Include edge events >= minimum duration for all metrics',
            'metrics': calculate_metrics_with_threshold(events, total_frames, fps)
        }
    }
    
    # Calculate impact
    edge_event_impact = {
        'edge_event_characteristics': analyze_edge_events(events),
        'metric_reliability': primary_metrics['duration_metric_reliability'],
        'frequency_impact': 'none (all events always included)',
        'duration_impact': calculate_duration_bias(events),
        'recommended_approach': 'dual metric strategy'
    }
    
    return primary_metrics, alternative_strategies, edge_event_impact

def compare_groups(group1_data, group2_data, test_type='auto'):
    """
    Statistical comparison between genotypes
    
    For each metric:
    1. Test normality (Shapiro-Wilk)
    2. Apply appropriate test:
       - Normal: Welch's t-test
       - Non-normal: Mann-Whitney U
    3. Calculate effect size:
       - Normal: Cohen's d
       - Non-normal: Cliff's delta
    4. Apply FDR correction for multiple comparisons
    """
```

**Visualizations** (all saved as high-resolution PDFs, extending existing plotting functions):
1. **Grooming Raster Plot** (enhanced from existing code)
2. **Cumulative Grooming Curves** (new)
3. **Bout Duration Distributions** (enhanced from existing box plots)
4. **Effect Size Forest Plot** (comparison mode only, new)
5. **Edge Event Sensitivity Analysis** (new)

**Output Structure**:
```
wild_type/
├── [existing files]
└── results/
    ├── summary_statistics.csv
    ├── individual_fly_metrics.csv
    ├── edge_event_sensitivity.csv
    ├── figures/
    │   ├── raster_plot.pdf
    │   ├── cumulative_curves.pdf
    │   ├── bout_distributions.pdf
    │   └── edge_sensitivity_analysis.pdf
    └── analysis_report.html
```

For comparison analysis:
```
comparison_wild_type_vs_mutant_X/
├── summary_statistics.csv
├── pairwise_comparisons.csv
├── pairwise_comparisons_fdr_corrected.csv
├── individual_fly_metrics.csv
├── edge_event_sensitivity.csv
├── multiple_testing_report.txt
├── figures/
│   ├── raster_plot.pdf
│   ├── cumulative_curves.pdf
│   ├── bout_distributions.pdf
│   ├── effect_sizes.pdf
│   └── edge_sensitivity_analysis.pdf
└── full_report.html
```

## 4. Implementation Notes

### 4.1 Edge Cases and Error Handling
* Videos not exactly expected frames: Warn user, proceed with actual length, document in metadata
* No grooming detected: Report zero values, not missing (empty CSV with just header)
* All frames grooming: Handle inter-bout interval as undefined
* Missing expected files: Clear error message indicating which file is missing and expected location
* Separator frames: Exclude from analysis (tracked in metadata)
* Odd number of frame entries: Error with message about unpaired start/stop
* Non-increasing frame numbers: Error indicating position of problematic frame
* Invalid grooming events: Error if stop frame < start frame
* Events beyond video length: Warn and clip to video bounds
* Overlapping events: Error with details about which events overlap
* FDR correction with single metric: Skip correction, report raw p-values only
* Empty genetic background: Error if folder name is empty or contains only special characters
* Variable frame counts: All statistics scale appropriately with actual frame count

### 4.2 Performance Considerations
* Use numpy for efficient array operations
* Implement progress bars for long operations (reuse from existing code)
* Save intermediate results for debugging
* Cache frame mapping for large datasets

### 4.3 Validation Checks
* Ensure frame numbers are continuous
* Verify scoring is binary (0 or 1 only)
* Check that reconstructed event times match original
* Validate that group_info.json matches the folder name
* Verify separator frames are properly excluded from analysis
* Confirm actual frame counts match expected (warn if different)

### 4.4 User Interface
* Clear command-line arguments with help text
* Informative progress messages (using existing logging framework)
* Detailed error messages with solutions
* Automatic detection of analysis mode based on arguments
* Display genetic background prominently in all outputs

### 4.5 Dependencies and Requirements
**Python Version**: 3.8+

**Required Libraries** (extends existing environment.yml):
```yaml
# Add to existing environment.yml from fly_behavior_analysis repo
dependencies:
  # Existing dependencies maintained
  - numpy>=1.20.0
  - pandas>=1.3.0
  - matplotlib>=3.4.0
  # New additions
  - scipy>=1.7.0           # Statistical tests
  - seaborn>=0.11.0        # Statistical visualizations
  - tifffile>=2021.7.0     # TIF file I/O
  - scikit-image>=0.18.0   # Image processing
  - statsmodels>=0.12.0    # Multiple testing correction
  - tqdm>=4.62.0           # Progress bars (if not already included)
```

**Project Structure**:
```
grooming_analysis/
├── pilot_grooming_optimizer.py
├── video_sampler.py
├── grooming_statistical_analyzer.py
├── fly_behavior_analysis/        # Existing repository as submodule
│   ├── main.py
│   ├── process_csv.py
│   ├── generate_timeline.py
│   └── ...
└── requirements.txt
```

**Installation**:
```bash
# Clone with submodule
git clone --recurse-submodules <your-repo-url>

# Or if already cloned
git submodule init
git submodule update

# Install dependencies
conda env create -f environment.yml
conda activate fly_analysis
```

## 5. Example Workflow

**Important**: Keep `sampling_metadata.csv` and `group_info.json` (created by `video_sampler.py`) together with your scored data file for proper statistical analysis.

```bash
# Step 1: Optimize parameters from pilot data (13 flies, full 9000 frames each)
python pilot_grooming_optimizer.py \
    --data-dir ./pilot_data/ \
    --output ./optimization_results.json \
    --total-frames 9000

# Review output to see:
# - Recommended window size: 300 frames
# - Recommended sampling: 7.5%
# - Minimum edge duration: 15 frames
# - Bout duration stats: mean 2.3s, min 0.5s, max 12.1s
# - Edge event impact: ~8% of events will be edge events
# - Complete event capture: 92% of events fully within windows
# - Metric strategy: Dual approach (all events for frequency, complete for duration)

# Step 2: Generate sampled videos for each genetic background
# Wild-type flies
python video_sampler.py \
    --video-dir ./wild_type/ \
    --window-size 300 \
    --sampling-percent 7.5 \
    --seed 42 \
    --total-frames 9000

# Mutant flies  
python video_sampler.py \
    --video-dir ./mutant_X/ \
    --window-size 300 \
    --sampling-percent 7.5 \
    --seed 42 \
    --total-frames 9000

# Step 3: Manual scoring
# User watches sampled_video.tif and creates event-based CSV file
# Record frame numbers where grooming starts and stops
# Save as CSV file with 'Frame' column header
# Example scored_sampled.csv:
#   Frame
#   120    (grooming starts)
#   245    (grooming stops)
#   580    (grooming starts)
#   672    (grooming stops)

# Step 4: Statistical analysis
# Single group analysis
python grooming_statistical_analyzer.py \
    --scored-data ./wild_type/scored_sampled.csv \
    --metadata ./wild_type/sampling_metadata.csv \
    --group-info ./wild_type/group_info.json \
    --output-dir ./wild_type/results/

# Comparison analysis
python grooming_statistical_analyzer.py \
    --scored-data-1 ./wild_type/scored_sampled.csv \
    --metadata-1 ./wild_type/sampling_metadata.csv \
    --group-info-1 ./wild_type/group_info.json \
    --scored-data-2 ./mutant_X/scored_sampled.csv \
    --metadata-2 ./mutant_X/sampling_metadata.csv \
    --group-info-2 ./mutant_X/group_info.json \
    --output-dir ./comparison_results/ \
    --fdr-method benjamini-hochberg

# For different video lengths (e.g., 12000 frames at 30 fps = 6.67 minutes)
python pilot_grooming_optimizer.py \
    --data-dir ./pilot_data_long/ \
    --output ./optimization_results_12k.json \
    --total-frames 12000
```

This pipeline will reduce ~13 hours of scoring to ~1 hour while maintaining statistical power for detecting biologically meaningful differences in grooming behavior, with appropriate controls for multiple testing, comprehensive edge event analysis, and full transparency about sampling trade-offs.

## 6. Key Advantages of This Design

1. **Statistical Rigor**: Clean, defensible statistics with proper FDR control
2. **Optimized Accuracy**: Dual approach maximizes accuracy - ALL events for frequency metrics, complete events only for unbiased duration metrics
3. **Edge Event Transparency**: Full analysis of edge event impact with multiple handling strategies
4. **Flexibility**: Adapts to different video lengths and experimental protocols
5. **Code Reuse**: Leverages tested, validated functions from existing repository
6. **User-Friendly**: Clear folder-based organization and automated metadata tracking
7. **Reproducibility**: Random seeds and comprehensive metadata ensure reproducible results
8. **Comprehensive Output**: Both detailed CSVs for further analysis and publication-ready figures
9. **Reliability Indicators**: Clear reporting of metric reliability based on available data

The design prioritizes practical usability and statistical validity while maximizing the use of available data through intelligent edge event handling.