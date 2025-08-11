# Drosophila Grooming Analysis Pipeline: Design Document (v3)

## 1. Project Overview

### 1.1 Problem Statement
Manual scoring of Drosophila grooming behavior in time-lapse microscopy videos is extremely time-intensive (1 hour per 9000-frame video). With insufficient camera resolution for automated neural network classification, we need an optimized manual sampling strategy that maintains statistical power while dramatically reducing analysis time.

### 1.2 Solution Design
A three-stage pipeline that:
1. **Determines optimal sampling parameters through comprehensive multi-dimensional analysis of pilot data**
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
* Multi-dimensional sampling optimization algorithms
* Comprehensive statistical power analysis
* Advanced bias assessment and mitigation
* Interactive decision support visualizations
* Cross-validation and bootstrap validation frameworks

## 2. Design Decisions and Rationale

### 2.1 Multi-Dimensional Optimization Framework
**Decision**: Implement comprehensive decision matrix across five key behavioral dimensions
**Rationale**: 
* Different grooming metrics respond differently to sampling parameters
* Bias and statistical power vary independently across metrics
* Single-metric optimization may miss critical trade-offs
* Heat map visualizations enable informed parameter selection

### 2.2 Comprehensive Pilot Analysis Requirements
**Decision**: Require complete evaluation of all metrics, biases, and sampling strategies in pilot phase
**Rationale**:
* Sampling parameters significantly impact statistical conclusions
* Bias assessment prevents systematic errors in downstream analysis
* Power analysis ensures adequate sensitivity for biological effects
* Alternative sampling strategies may outperform uniform random sampling

### 2.3 Statistical Rigor Framework
**Decision**: Implement formal validation with cross-validation, bootstrap analysis, and multiple testing correction
**Rationale**:
* Prevents overfitting of sampling parameters to specific pilot data
* Provides robust estimates of error rates and statistical power
* Ensures reproducible and defensible methodology for publication

### 2.4 Sampling Strategy Selection
**Decision**: Use non-overlapping windows with uniform random sampling as baseline, but evaluate alternatives
**Rationale**: 
* Non-overlapping windows prevent double-counting events
* Uniform sampling ensures unbiased temporal coverage
* Random selection maintains statistical validity for inference
* Alternative strategies tested systematically for potential improvements

### 2.5 Window Size Selection
**Decision**: Choose window size that evenly divides total frames AND captures 80% of complete grooming events
**Rationale**: 
* Even division prevents partial window complications
* 80% capture ensures most events are fully observed
* For 9000 frames: viable sizes include 100, 150, 180, 200, 225, 250, 300, 360, 450, 500, 600, 750, 900
* Algorithm adapts to different total frame counts

### 2.6 Edge Event Handling and Trade-offs

#### 2.6.1 The Edge Event Problem
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

#### 2.6.2 Dual Metric Strategy
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

### 2.7 Bootstrap Validation
**Decision**: 10,000 bootstrap iterations per parameter combination
**Rationale**: 
* Provides robust confidence intervals
* Accounts for event clustering and temporal variability
* Enables optimization across multiple criteria

### 2.8 Multiple Testing Correction
**Decision**: Apply False Discovery Rate (FDR) correction using Benjamini-Hochberg procedure
**Rationale**: When comparing genotypes across multiple grooming metrics, we perform multiple statistical tests that require correction to control Type I error inflation.

### 2.9 Folder-Based Group Organization
**Decision**: Use folder names as group identifiers for genetic background
**Rationale**: 
* Eliminates manual CSV creation and potential errors
* Enforces organized data structure
* Simplifies user interface
* Maintains clear separation between experimental groups

### 2.10 Configurable Frame Count
**Decision**: Make total frame count a configurable parameter (default: 9000)
**Rationale**: 
* Different experimental protocols may use different video lengths
* Algorithm automatically adapts window size recommendations
* Maintains flexibility for future experiments

## 3. Script Specifications

### 3.1 Script 1: `pilot_grooming_optimizer.py`

**Purpose**: Comprehensive multi-dimensional analysis of pilot data to determine optimal sampling strategy across all behavioral metrics

**Integration with existing code**: 
- Imports `process_csv`, `generate_timeline`, `generate_event_list` from existing repository
- Extends with sophisticated optimization algorithms and statistical analysis
- Adds comprehensive visualization suite for decision support

#### 3.1.1 Input Requirements

**Pilot Data Structure**:
```python
pilot_requirements = {
    'minimum_sample_size': 15,  # flies per genotype
    'recommended_sample_size': 25,  # for robust parameter estimation
    'video_length': 'Full length (default 9000 frames)',
    'annotation_completeness': 'Frame-by-frame grooming annotation for ALL videos',
    'genotype_diversity': 'Include control and experimental genotypes if available',
    'quality_control': 'Manual validation of all annotations'
}
```

**Input Files**:
* Directory path containing CSV files with event-based annotations
* Each file: alternating start/stop frame numbers (even number of rows)
* Format: Single column labeled 'Frame'
* One file per video (any filename accepted)
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

#### 3.1.2 Core Analysis Framework

```python
def main(data_directory, output_path, total_frames=9000, analysis_mode='comprehensive'):
    """
    Comprehensive pilot analysis pipeline
    
    Analysis Components:
    1. Baseline behavior characterization
    2. Multi-dimensional parameter optimization 
    3. Statistical power analysis across effect sizes
    4. Bias assessment (temporal, duration threshold, sampling strategy)
    5. Error rate estimation via cross-validation
    6. Decision matrix generation with visualizations
    7. Sensitivity analysis and robustness testing
    """
    
    # Import from existing repository
    from fly_behavior_analysis import (
        process_csv,
        generate_timeline,
        generate_event_list,
        create_visualizations
    )
    
    # Core data structures
    BehaviorMetrics = {
        'event_frequency': 'Events per minute',
        'event_duration': 'Mean bout duration (seconds)', 
        'event_percentage': 'Percentage of time grooming',
        'bout_duration': 'Individual bout lengths',
        'fragmentation_index': 'Coefficient of variation of bout durations'
    }
    
    SamplingParameters = {
        'window_size': list(get_viable_window_sizes(total_frames)),
        'sampling_rate': [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30],
        'min_duration': [5, 10, 15, 20, 25, 30]  # frames
    }
    
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

#### 3.1.3 Multi-Dimensional Decision Matrix

```python
class ComprehensiveDecisionMatrix:
    """
    Multi-dimensional optimization and decision support
    """
    
    def __init__(self, pilot_data, total_frames=9000):
        self.pilot_data = pilot_data
        self.total_frames = total_frames
        self.metrics = ['event_frequency', 'event_duration', 'event_percentage', 
                       'bout_duration', 'fragmentation_index']
        
    def generate_decision_matrix(self):
        """
        Create comprehensive parameter evaluation across all dimensions
        
        Returns:
        - optimization_results: 5D array [metric][window][sampling][min_dur][criterion]
        - statistical_power: power analysis for each combination
        - bias_assessment: bias estimates across all dimensions
        - error_rates: FPR/FNR estimates via cross-validation
        - recommendations: ranked parameter combinations
        """
        
        # Initialize results structure
        results = {}
        for metric in self.metrics:
            results[metric] = {}
            
        # Multi-dimensional parameter sweep
        for window_size in self.viable_window_sizes:
            for sampling_rate in self.sampling_rates:
                for min_duration in self.min_durations:
                    
                    # Core evaluation for this parameter combination
                    eval_results = self.evaluate_parameter_combination(
                        window_size, sampling_rate, min_duration
                    )
                    
                    # Store results for each metric
                    for metric in self.metrics:
                        if window_size not in results[metric]:
                            results[metric][window_size] = {}
                        if sampling_rate not in results[metric][window_size]:
                            results[metric][window_size][sampling_rate] = {}
                        
                        results[metric][window_size][sampling_rate][min_duration] = eval_results[metric]
        
        return results
    
    def evaluate_parameter_combination(self, window_size, sampling_rate, min_duration):
        """
        Comprehensive evaluation of single parameter combination
        
        Returns evaluation across all metrics:
        - bias_estimate: systematic error vs ground truth
        - variance_estimate: precision of metric estimation  
        - statistical_power: minimum detectable effect size
        - error_rates: false positive/negative rates
        - efficiency: time savings vs accuracy trade-off
        """
        
        evaluation = {}
        
        for metric in self.metrics:
            # Cross-validation framework
            cv_results = self.cross_validate_sampling(
                metric, window_size, sampling_rate, min_duration
            )
            
            # Statistical power analysis
            power_analysis = self.estimate_statistical_power(
                metric, window_size, sampling_rate, min_duration
            )
            
            # Bias assessment
            bias_analysis = self.assess_metric_bias(
                metric, window_size, sampling_rate, min_duration
            )
            
            evaluation[metric] = {
                'bias_estimate': bias_analysis['mean_bias'],
                'bias_variance': bias_analysis['bias_variance'],
                'variance_inflation': cv_results['variance_ratio'],
                'min_detectable_effect': power_analysis['min_detectable_d'],
                'statistical_power_d08': power_analysis['power_at_d08'],
                'false_positive_rate': cv_results['fpr'],
                'false_negative_rate': cv_results['fnr'],
                'time_efficiency': calculate_time_savings(window_size, sampling_rate),
                'reliability_score': self.calculate_reliability_score(cv_results, bias_analysis),
                'edge_event_proportion': bias_analysis['edge_event_rate']
            }
        
        return evaluation
```

#### 3.1.4 Statistical Power Analysis

```python
def estimate_statistical_power(self, metric, window_size, sampling_rate, min_duration):
    """
    Estimate statistical power for detecting biologically meaningful differences
    
    Analysis approach:
    1. Estimate metric variance from pilot data
    2. Calculate effective sample size after sampling
    3. Bootstrap simulation of sampling process
    4. Power analysis across range of effect sizes
    """
    
    # Extract baseline metric distribution
    pilot_values = self.calculate_pilot_metric_values(metric)
    baseline_variance = np.var(pilot_values)
    
    # Simulate sampling process
    sampling_variances = []
    effective_samples = []
    
    for bootstrap_iter in range(1000):
        sampled_data = self.simulate_sampling(
            self.pilot_data, window_size, sampling_rate, min_duration
        )
        sampled_values = self.calculate_metric_from_sampled(sampled_data, metric)
        
        sampling_variances.append(np.var(sampled_values))
        effective_samples.append(len(sampled_values))
    
    # Power analysis
    mean_sampling_variance = np.mean(sampling_variances)
    mean_effective_n = np.mean(effective_samples)
    
    # Calculate minimum detectable effect sizes
    effect_sizes = np.linspace(0.2, 2.0, 50)
    power_curve = []
    
    for d in effect_sizes:
        power = self.calculate_statistical_power_ttest(
            d, mean_effective_n, mean_sampling_variance, alpha=0.05
        )
        power_curve.append(power)
    
    # Find minimum detectable effect at 80% power
    min_detectable_d = np.interp(0.80, power_curve, effect_sizes)
    power_at_d08 = np.interp(0.8, effect_sizes, power_curve)
    
    return {
        'min_detectable_d': min_detectable_d,
        'power_at_d08': power_at_d08,
        'power_curve': list(zip(effect_sizes, power_curve)),
        'effective_sample_size': mean_effective_n,
        'variance_inflation': mean_sampling_variance / baseline_variance
    }

def calculate_statistical_power_ttest(self, effect_size, n, variance, alpha=0.05):
    """
    Calculate statistical power for two-sample t-test
    
    Uses non-central t-distribution for exact power calculation
    """
    from scipy import stats
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n / 2)
    
    # Critical value for two-tailed test
    df = 2 * n - 2
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Power calculation using non-central t-distribution
    power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
    
    return power
```

#### 3.1.5 Bias Assessment Framework

```python
def assess_comprehensive_bias(self):
    """
    Multi-dimensional bias assessment
    
    Bias Types Evaluated:
    1. Temporal bias: Does uniform random sampling miss time-dependent patterns?
    2. Duration threshold bias: How does minimum duration filtering affect metrics?
    3. Edge event bias: Impact of partial event capture at window boundaries
    4. Sample size bias: Bias as function of effective sample size
    """
    
    bias_results = {
        'temporal_bias': self.assess_temporal_bias(),
        'duration_threshold_bias': self.assess_duration_threshold_bias(),
        'edge_event_bias': self.assess_edge_event_bias(),
        'sampling_strategy_comparison': self.compare_sampling_strategies()
    }
    
    return bias_results

def assess_temporal_bias(self):
    """
    Test whether uniform random sampling introduces temporal bias
    """
    
    # Divide videos into temporal blocks
    n_blocks = 5
    block_size = self.total_frames // n_blocks
    
    temporal_analysis = {}
    
    for metric in self.metrics:
        # Calculate metric in each temporal block
        block_values = []
        for block_idx in range(n_blocks):
            start_frame = block_idx * block_size
            end_frame = (block_idx + 1) * block_size
            
            block_data = self.extract_temporal_block(
                self.pilot_data, start_frame, end_frame
            )
            block_metric = self.calculate_metric_from_data(block_data, metric)
            block_values.append(block_metric)
        
        # Test for temporal trends
        temporal_trend_p = stats.kendalltau(range(n_blocks), block_values)[1]
        temporal_variance = np.var(block_values)
        
        # Compare uniform random vs stratified temporal sampling
        uniform_bias = self.simulate_uniform_sampling_bias(metric)
        stratified_bias = self.simulate_stratified_temporal_bias(metric)
        
        temporal_analysis[metric] = {
            'temporal_trend_p': temporal_trend_p,
            'temporal_variance': temporal_variance,
            'uniform_sampling_bias': uniform_bias,
            'stratified_sampling_bias': stratified_bias,
            'bias_reduction_stratified': uniform_bias - stratified_bias
        }
    
    return temporal_analysis

def assess_duration_threshold_bias(self):
    """
    Quantify bias introduced by minimum duration filtering
    
    For each metric:
    1. Calculate true value (no filtering)
    2. Calculate filtered value for each threshold
    3. Quantify bias = (filtered - true) / true
    4. Test bias consistency across genotypes
    """
    
    bias_analysis = {}
    for metric in self.metrics:
        true_values = self.calculate_metric_no_filter(self.pilot_data, metric)
        
        bias_analysis[metric] = {}
        for min_dur in self.min_durations:
            filtered_values = self.calculate_metric_with_filter(
                self.pilot_data, metric, min_dur
            )
            
            bias = (filtered_values - true_values) / true_values
            bias_analysis[metric][min_dur] = {
                'mean_bias': np.mean(bias),
                'bias_variance': np.var(bias),
                'genotype_consistency': self.test_bias_consistency(
                    bias, self.pilot_data.genotype
                )
            }
    
    return bias_analysis

def compare_sampling_strategies(self):
    """
    Compare multiple sampling approaches
    """
    
    sampling_strategies = {
        'uniform_random': self.simulate_uniform_random_sampling,
        'stratified_temporal': self.simulate_stratified_temporal_sampling,
        'systematic': self.simulate_systematic_sampling,
        'adaptive_density': self.simulate_adaptive_density_sampling,
        'clustered': self.simulate_clustered_sampling
    }
    
    strategy_comparison = {}
    
    for strategy_name, strategy_func in sampling_strategies.items():
        strategy_results = {}
        
        for metric in self.metrics:
            # Evaluate strategy across parameter combinations
            bias_estimates = []
            variance_estimates = []
            
            for window_size in self.viable_window_sizes[:3]:  # Sample for efficiency
                for sampling_rate in [0.10, 0.15, 0.20]:
                    
                    bias, variance = strategy_func(metric, window_size, sampling_rate)
                    bias_estimates.append(bias)
                    variance_estimates.append(variance)
            
            strategy_results[metric] = {
                'mean_bias': np.mean(bias_estimates),
                'mean_variance': np.mean(variance_estimates),
                'bias_stability': np.std(bias_estimates),
                'variance_stability': np.std(variance_estimates)
            }
        
        strategy_comparison[strategy_name] = strategy_results
    
    return strategy_comparison
```

#### 3.1.6 Cross-Validation Error Rate Estimation

```python
def cross_validate_sampling(self, metric, window_size, sampling_rate, min_duration, cv_folds=5):
    """
    Estimate false positive/negative rates via cross-validation
    
    Approach:
    1. Split pilot data into CV folds
    2. For each fold: use full data as ground truth, sampled data as test
    3. Compare detected differences vs ground truth
    4. Aggregate error rates across folds
    """
    
    cv_results = []
    
    # Cross-validation loop
    for fold_idx in range(cv_folds):
        fold_data = self.get_cv_fold(fold_idx, cv_folds)
        
        # Ground truth: analysis on full data
        ground_truth = self.analyze_full_data(fold_data, metric)
        
        # Test: analysis on sampled data
        sampled_data = self.simulate_sampling(
            fold_data, window_size, sampling_rate, min_duration
        )
        sampled_result = self.analyze_sampled_data(sampled_data, metric)
        
        # Calculate error rates
        fold_errors = self.calculate_detection_errors(ground_truth, sampled_result)
        cv_results.append(fold_errors)
    
    # Aggregate across folds
    aggregated_results = {
        'fpr': np.mean([r['fpr'] for r in cv_results]),
        'fnr': np.mean([r['fnr'] for r in cv_results]),
        'variance_ratio': np.mean([r['variance_ratio'] for r in cv_results]),
        'bias_estimate': np.mean([r['bias'] for r in cv_results]),
        'confidence_interval': self.calculate_cv_confidence_intervals(cv_results)
    }
    
    return aggregated_results

def calculate_detection_errors(self, ground_truth, sampled_result):
    """
    Calculate false positive/negative rates for metric detection
    
    For continuous metrics, "detection" means:
    - Detecting difference between genotypes (if available)
    - Detecting presence of grooming behavior
    - Detecting significant deviation from baseline
    """
    
    # Define "positive" detection criteria
    if 'genotype_difference' in ground_truth:
        # Compare genotype difference detection
        true_difference = ground_truth['genotype_difference']['significant']
        detected_difference = sampled_result['genotype_difference']['significant']
        
        # Calculate error rates
        if true_difference and detected_difference:
            outcome = 'true_positive'
        elif true_difference and not detected_difference:
            outcome = 'false_negative'
        elif not true_difference and detected_difference:
            outcome = 'false_positive'
        else:
            outcome = 'true_negative'
    else:
        # Single group analysis - focus on bias and variance
        true_value = ground_truth['metric_value']
        detected_value = sampled_result['metric_value']
        
        bias = (detected_value - true_value) / true_value
        variance_ratio = sampled_result['variance'] / ground_truth['variance']
        
        outcome = 'single_group'
    
    return {
        'outcome': outcome,
        'fpr': 1.0 if outcome == 'false_positive' else 0.0,
        'fnr': 1.0 if outcome == 'false_negative' else 0.0,
        'bias': bias if outcome == 'single_group' else 0.0,
        'variance_ratio': variance_ratio if outcome == 'single_group' else 1.0
    }
```

#### 3.1.7 Comprehensive Visualization Suite

```python
def create_decision_visualizations(self, optimization_results):
    """
    Generate comprehensive visualization suite for parameter selection
    
    Visualizations:
    1. Multi-dimensional heat maps (metric × window_size × sampling_rate)
    2. Pareto frontier plots (accuracy vs efficiency)
    3. Statistical power curves
    4. Bias assessment plots
    5. Interactive decision dashboard
    """
    
    # 1. Heat map grid for each metric
    self.create_metric_heatmaps(optimization_results)
    
    # 2. Pareto frontier analysis
    self.create_pareto_frontiers(optimization_results)
    
    # 3. Statistical power visualization
    self.create_power_analysis_plots(optimization_results)
    
    # 4. Bias assessment plots
    self.create_bias_assessment_plots()
    
    # 5. Interactive dashboard
    self.create_interactive_dashboard(optimization_results)

def create_metric_heatmaps(self, optimization_results):
    """
    Create heat map grid: window_size × sampling_rate with min_duration subplots
    """
    
    for metric in self.metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Parameter Optimization: {metric.replace("_", " ").title()}', fontsize=16)
        
        for idx, min_duration in enumerate([5, 10, 15, 20, 25, 30]):
            row, col = idx // 3, idx % 3
            
            # Extract heat map matrix
            heatmap_data = self.extract_heatmap_matrix(
                optimization_results[metric], min_duration, criterion='reliability_score'
            )
            
            # Create heat map
            sns.heatmap(
                heatmap_data,
                ax=axes[row, col],
                xticklabels=[f'{r:.1%}' for r in self.sampling_rates],
                yticklabels=self.viable_window_sizes,
                cmap='RdYlGn',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Reliability Score'},
                annot=True, fmt='.2f'
            )
            
            axes[row, col].set_title(f'Min Duration: {min_duration} frames')
            axes[row, col].set_xlabel('Sampling Rate')
            axes[row, col].set_ylabel('Window Size (frames)')
        
        plt.tight_layout()
        plt.savefig(f'heatmap_{metric}.pdf', dpi=300, bbox_inches='tight')

def create_pareto_frontiers(self, optimization_results):
    """
    Multi-objective optimization: accuracy vs efficiency trade-offs
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, metric in enumerate(self.metrics):
        row, col = idx // 3, idx % 3
        
        # Extract efficiency and accuracy for all parameter combinations
        efficiency_scores = []
        accuracy_scores = []
        param_labels = []
        
        for window_size in self.viable_window_sizes:
            for sampling_rate in self.sampling_rates:
                for min_duration in [15]:  # Representative value
                    
                    result = optimization_results[metric][window_size][sampling_rate][min_duration]
                    
                    efficiency = result['time_efficiency']
                    accuracy = 1 - result['bias_estimate']**2 - result['variance_inflation']
                    
                    efficiency_scores.append(efficiency)
                    accuracy_scores.append(accuracy)
                    param_labels.append(f'W{window_size}_S{sampling_rate:.0%}_D{min_duration}')
        
        # Plot scatter with Pareto frontier
        axes[row, col].scatter(efficiency_scores, accuracy_scores, alpha=0.6)
        
        # Identify and plot Pareto frontier
        pareto_indices = self.find_pareto_frontier(efficiency_scores, accuracy_scores)
        pareto_efficiency = [efficiency_scores[i] for i in pareto_indices]
        pareto_accuracy = [accuracy_scores[i] for i in pareto_indices]
        
        axes[row, col].plot(pareto_efficiency, pareto_accuracy, 'r-', linewidth=2, 
                           label='Pareto Frontier')
        
        axes[row, col].set_xlabel('Time Efficiency')
        axes[row, col].set_ylabel('Accuracy Score')
        axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('pareto_frontiers.pdf', dpi=300, bbox_inches='tight')

def create_interactive_dashboard(self, optimization_results):
    """
    Create interactive Plotly dashboard for parameter exploration
    """
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html, Input, Output
    
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Drosophila Grooming Sampling Optimization Dashboard"),
        
        html.Div([
            html.Label("Select Metric:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[{'label': metric.replace('_', ' ').title(), 'value': metric} 
                        for metric in self.metrics],
                value=self.metrics[0]
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Select Criterion:"),
            dcc.Dropdown(
                id='criterion-dropdown',
                options=[
                    {'label': 'Reliability Score', 'value': 'reliability_score'},
                    {'label': 'Statistical Power', 'value': 'statistical_power_d08'},
                    {'label': 'Bias Estimate', 'value': 'bias_estimate'},
                    {'label': 'Time Efficiency', 'value': 'time_efficiency'}
                ],
                value='reliability_score'
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Minimum Duration (frames):"),
            dcc.Slider(
                id='min-duration-slider',
                min=5, max=30, step=5, value=15,
                marks={i: str(i) for i in range(5, 31, 5)}
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        dcc.Graph(id='heatmap-plot'),
        dcc.Graph(id='power-curve-plot')
    ])
    
    @app.callback(
        [Output('heatmap-plot', 'figure'),
         Output('power-curve-plot', 'figure')],
        [Input('metric-dropdown', 'value'),
         Input('criterion-dropdown', 'value'),
         Input('min-duration-slider', 'value')]
    )
    def update_plots(selected_metric, selected_criterion, min_duration):
        # Generate heatmap
        heatmap_fig = self.create_interactive_heatmap(
            optimization_results, selected_metric, selected_criterion, min_duration
        )
        
        # Generate power curve
        power_fig = self.create_power_curve_plot(
            optimization_results, selected_metric, min_duration
        )
        
        return heatmap_fig, power_fig
    
    return app
```

#### 3.1.8 Final Recommendations Engine

```python
def generate_recommendations(self, optimization_results, bias_assessment, power_analysis):
    """
    Generate ranked recommendations based on multi-criteria decision analysis
    
    Criteria:
    1. Statistical power (≥80% for d=0.8)
    2. Bias magnitude (<15% for all metrics)
    3. Error rates (FPR <5%, FNR <20%)
    4. Time efficiency (maximize sampling reduction)
    5. Robustness across metrics
    """
    
    # Score all parameter combinations
    parameter_scores = []
    
    for window_size in self.viable_window_sizes:
        for sampling_rate in self.sampling_rates:
            for min_duration in self.min_durations:
                
                # Calculate composite score
                composite_score = self.calculate_composite_score(
                    optimization_results, window_size, sampling_rate, min_duration
                )
                
                parameter_scores.append({
                    'window_size': window_size,
                    'sampling_rate': sampling_rate,
                    'min_duration': min_duration,
                    'composite_score': composite_score['total_score'],
                    'individual_scores': composite_score['component_scores'],
                    'meets_constraints': composite_score['constraint_satisfaction'],
                    'recommendation_tier': self.assign_recommendation_tier(composite_score)
                })
    
    # Sort by composite score
    ranked_recommendations = sorted(
        parameter_scores, key=lambda x: x['composite_score'], reverse=True
    )
    
    return {
        'top_recommendation': ranked_recommendations[0],
        'alternative_recommendations': ranked_recommendations[1:6],
        'all_evaluations': ranked_recommendations,
        'recommendation_rationale': self.generate_rationale(ranked_recommendations[0]),
        'sensitivity_analysis': self.perform_sensitivity_analysis(ranked_recommendations[:10])
    }

def calculate_composite_score(self, optimization_results, window_size, sampling_rate, min_duration):
    """
    Multi-criteria scoring function
    """
    
    weights = {
        'statistical_power': 0.25,
        'bias_magnitude': 0.25,
        'error_rates': 0.20,
        'time_efficiency': 0.15,
        'robustness': 0.15
    }
    
    component_scores = {}
    constraint_violations = []
    
    # Aggregate across metrics
    metric_results = []
    for metric in self.metrics:
        result = optimization_results[metric][window_size][sampling_rate][min_duration]
        metric_results.append(result)
    
    # 1. Statistical power score
    power_scores = [r['statistical_power_d08'] for r in metric_results]
    component_scores['statistical_power'] = np.mean(power_scores)
    if np.min(power_scores) < 0.8:
        constraint_violations.append('insufficient_power')
    
    # 2. Bias magnitude score (lower bias = higher score)
    bias_scores = [1 - abs(r['bias_estimate']) for r in metric_results]
    component_scores['bias_magnitude'] = np.mean(bias_scores)
    if np.max([abs(r['bias_estimate']) for r in metric_results]) > 0.15:
        constraint_violations.append('excessive_bias')
    
    # 3. Error rates score
    fpr_scores = [1 - r['false_positive_rate'] for r in metric_results]
    fnr_scores = [1 - r['false_negative_rate'] for r in metric_results]
    component_scores['error_rates'] = (np.mean(fpr_scores) + np.mean(fnr_scores)) / 2
    
    if np.max([r['false_positive_rate'] for r in metric_results]) > 0.05:
        constraint_violations.append('high_fpr')
    if np.max([r['false_negative_rate'] for r in metric_results]) > 0.20:
        constraint_violations.append('high_fnr')
    
    # 4. Time efficiency score
    component_scores['time_efficiency'] = metric_results[0]['time_efficiency']  # Same for all metrics
    
    # 5. Robustness score (consistency across metrics)
    reliability_scores = [r['reliability_score'] for r in metric_results]
    component_scores['robustness'] = 1 - np.std(reliability_scores)
    
    # Calculate weighted total
    total_score = sum(weights[criterion] * score for criterion, score in component_scores.items())
    
    return {
        'total_score': total_score,
        'component_scores': component_scores,
        'constraint_satisfaction': len(constraint_violations) == 0,
        'constraint_violations': constraint_violations
    }
```

#### 3.1.9 Key Functions

**Key Functions**:
1. `load_pilot_data(directory)` - Load all CSV files using existing `process_csv`
2. `analyze_bout_durations(all_events)` - Generate duration distribution statistics
3. `get_viable_window_sizes(total_frames, min_size=None)` - Find all divisors of total_frames
4. `generate_decision_matrix()` - Create comprehensive parameter evaluation
5. `estimate_statistical_power()` - Power analysis for detecting biological effects
6. `assess_comprehensive_bias()` - Multi-dimensional bias assessment
7. `cross_validate_sampling()` - Error rate estimation via CV
8. `create_decision_visualizations()` - Heat maps and interactive dashboards
9. `generate_recommendations()` - Ranked parameter combinations with rationale

#### 3.1.10 Output Structure

**JSON Output** (`optimization_results.json`):
```json
{
    "pilot_data_summary": {
        "n_videos": 25,
        "total_frames": 9000,
        "baseline_behavior_statistics": {
            "mean_event_frequency": 2.3,
            "mean_bout_duration": 1.8,
            "grooming_percentage": 12.4
        },
        "genotype_information": {
            "genotypes_present": ["wild_type", "mutant_X"],
            "effect_sizes": {"event_frequency": 0.7, "bout_duration": 1.2}
        }
    },
    "decision_matrix": {
        "event_frequency": {
            "window_size_300": {
                "sampling_rate_0.15": {
                    "min_duration_15": {
                        "bias_estimate": -0.02,
                        "statistical_power_d08": 0.85,
                        "false_positive_rate": 0.03,
                        "false_negative_rate": 0.12,
                        "reliability_score": 0.91,
                        "time_efficiency": 0.85
                    }
                }
            }
        }
    },
    "recommendations": {
        "top_recommendation": {
            "window_size": 300,
            "sampling_rate": 0.15,
            "min_duration": 15,
            "composite_score": 0.87,
            "rationale": "Optimal balance of statistical power (85% for d=0.8), low bias (<3%), acceptable error rates, and high time efficiency (85% reduction)."
        },
        "alternative_recommendations": [
            {
                "window_size": 450,
                "sampling_rate": 0.12,
                "min_duration": 15,
                "composite_score": 0.84
            }
        ],
        "sensitivity_analysis": {
            "parameter_stability": "High - top recommendation maintains rank across bootstrap samples",
            "bias_sensitivity": "Low - recommendation robust to bias estimation uncertainty"
        }
    },
    "bias_assessment": {
        "temporal_bias": {
            "event_frequency": {
                "temporal_trend_p": 0.234,
                "uniform_sampling_bias": 0.02,
                "stratified_sampling_bias": 0.01,
                "bias_reduction_stratified": 0.01
            }
        },
        "duration_threshold_bias": {
            "event_frequency": {
                "min_duration_15": {
                    "mean_bias": -0.05,
                    "bias_variance": 0.02,
                    "genotype_consistency": true
                }
            }
        },
        "sampling_strategy_comparison": {
            "uniform_random": {"mean_bias": 0.02, "mean_variance": 0.15},
            "stratified_temporal": {"mean_bias": 0.01, "mean_variance": 0.14},
            "systematic": {"mean_bias": 0.03, "mean_variance": 0.13}
        }
    },
    "statistical_power_analysis": {
        "power_curves": {
            "event_frequency": {
                "effect_sizes": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "power_values": [0.23, 0.34, 0.47, 0.62, 0.74, 0.83, 0.89, 0.93, 0.96]
            }
        }
    },
    "methodology_validation": {
        "cross_validation_results": {
            "mean_fpr": 0.032,
            "mean_fnr": 0.121,
            "confidence_intervals": {"fpr": [0.018, 0.047], "fnr": [0.089, 0.154]}
        },
        "bootstrap_confidence_intervals": {
            "bias_estimates": [-0.035, -0.005],
            "power_estimates": [0.81, 0.89]
        }
    }
}
```

**PDF Report** (`pilot_optimization_report.pdf`):
- Executive summary with top recommendation
- Multi-dimensional heat map grids for all metrics
- Pareto frontier analysis
- Statistical power curves
- Bias assessment plots
- Sampling strategy comparison
- Detailed methodology and validation results

### 3.2 Script 2: `video_sampler.py`

**Purpose**: Generate sampled video containing only selected windows with visual separators

**Input**: 
* Directory containing TIF video files (folder name = genetic background)
* Window size (frames) from pilot optimization
* Sampling percentage from pilot optimization
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

def perform_sensitivity_analysis(events, total_frames, fps=30):
    """
    Analyze impact of edge events on different metrics
    
    Returns:
    - primary_metrics: using dual approach (all events for frequency, complete for duration)
    - alternative_strategies: dict of metrics using different approaches
    - edge_event_impact: detailed breakdown of edge effects
    """

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
* Parallel processing for bootstrap and cross-validation loops

### 4.3 Validation Checks
* Ensure frame numbers are continuous
* Verify scoring is binary (0 or 1 only)
* Check that reconstructed event times match original
* Validate that group_info.json matches the folder name
* Verify separator frames are properly excluded from analysis
* Confirm actual frame counts match expected (warn if different)

### 4.4 Computational Requirements
* Analysis runtime: ~2-4 hours for comprehensive pilot analysis
* Memory requirements: ~4-8 GB for large pilot datasets
* Parallel processing: Bootstrap and cross-validation loops parallelized
* Progress tracking: Detailed progress bars for long-running analyses

### 4.5 Quality Assurance
* Automated validation of all input data
* Cross-validation of optimization results
* Sensitivity analysis of key parameters
* Comprehensive error handling and logging

### 4.6 User Interface
* Clear command-line arguments with help text
* Informative progress messages (using existing logging framework)
* Detailed error messages with solutions
* Automatic detection of analysis mode based on arguments
* Display genetic background prominently in all outputs

### 4.7 Dependencies and Requirements

**Python Version**: 3.8+

**Required Libraries**:
```yaml
# Add to existing environment.yml from fly_behavior_analysis repo
dependencies:
  # Existing dependencies maintained
  - numpy>=1.20.0
  - pandas>=1.3.0
  - matplotlib>=3.4.0
  # New additions for comprehensive analysis
  - scipy>=1.7.0           # Statistical tests
  - seaborn>=0.11.0        # Statistical visualizations
  - tifffile>=2021.7.0     # TIF file I/O
  - scikit-image>=0.18.0   # Image processing
  - scikit-learn>=1.0.0    # Cross-validation, metrics
  - statsmodels>=0.13.0    # Multiple testing correction, advanced stats
  - plotly>=5.0.0          # Interactive visualizations
  - dash>=2.0.0            # Interactive dashboard
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
# Step 1: Comprehensive pilot analysis (enhanced version)
python pilot_grooming_optimizer.py \
    --data-dir ./pilot_data/ \
    --output ./optimization_results.json \
    --total-frames 9000 \
    --analysis-mode comprehensive \
    --cv-folds 5 \
    --bootstrap-iterations 10000 \
    --include-power-analysis \
    --compare-sampling-strategies

# Output interpretation:
# - Recommended parameters: window_size=300, sampling_rate=15%, min_duration=15
# - Statistical power: >80% for effect size d=0.8 across all metrics
# - Bias assessment: <5% bias for frequency metrics, <10% for duration metrics
# - Time savings: 85% reduction in analysis time
# - Confidence: High reliability across all behavioral dimensions
# - Alternative sampling strategies: uniform random performs best

# Step 2: Generate sampled videos for each genetic background
# Wild-type flies
python video_sampler.py \
    --video-dir ./wild_type/ \
    --window-size 300 \
    --sampling-percent 15.0 \
    --seed 42 \
    --total-frames 9000

# Mutant flies   
python video_sampler.py \
    --video-dir ./mutant_X/ \
    --window-size 300 \
    --sampling-percent 15.0 \
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

## 6. Key Advantages of Enhanced Design

1. **Rigorous Statistical Foundation**: Comprehensive validation with cross-validation, bootstrap analysis, and formal power calculations
2. **Multi-Dimensional Optimization**: Simultaneous optimization across all behavioral metrics prevents single-metric bias
3. **Bias Transparency**: Complete characterization of bias sources with quantitative assessment
4. **Alternative Strategy Evaluation**: Systematic comparison of sampling approaches beyond uniform random
5. **Decision Support**: Heat maps and Pareto frontiers enable informed parameter selection
6. **Robustness Testing**: Sensitivity analysis ensures parameter stability across conditions
7. **Publication Ready**: Comprehensive methodology suitable for peer review and publication
8. **Interactive Analysis**: Dashboard for real-time parameter exploration
9. **Comprehensive Documentation**: Full traceability of all analysis decisions
10. **Flexible Implementation**: Adapts to different experimental protocols and video lengths

This enhanced design transforms the pilot analysis from a basic parameter search into a sophisticated statistical optimization framework that ensures robust, defensible sampling strategies for behavioral genetics research. The multi-dimensional decision matrix provides unprecedented insight into the trade-offs between accuracy, efficiency, and statistical power across all behavioral dimensions of interest.