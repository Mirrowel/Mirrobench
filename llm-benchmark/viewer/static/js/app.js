/**
 * MirroBench - Vue.js 3 Application
 * Modern LLM Benchmark Viewer
 */

const { createApp } = Vue;

createApp({
    data() {
        return {
            // Navigation
            currentPage: 'individual',
            pages: [
                { id: 'individual', name: 'Individual Evaluation', icon: 'gauge' },
                { id: 'comparative', name: 'Comparative Judge', icon: 'git-compare' },
                { id: 'human', name: 'Human Judge', icon: 'user' },
                { id: 'authors-choice', name: "Author's Choice", icon: 'star' },
                { id: 'run-benchmark', name: 'Run Benchmark', icon: 'play-circle' }
            ],

            // UI State
            loading: false,
            showConfigModal: false,
            toasts: [],

            // Individual Evaluation Data
            leaderboard: [],
            modelDisplayNames: {},
            runs: [],
            questions: [],

            // Comparative Judge Data
            comparativeJobs: [],
            selectedJobResults: null,
            comparativeLoading: false,
            showStartJobModal: false,
            selectedRunsForComparison: [],

            // Human Judge Data
            humanJudgeSelectedRun: null,
            humanJudgeLeaderboard: [],
            humanJudgeQuestions: [],
            humanJudgeLoading: false,
            humanJudgeUnifiedLeaderboard: [],
            humanJudgeUnifiedLoading: false,
            showRatingSection: false,
            selectedQuestionForRating: null,
            currentRating: { score: 50, comment: '' },

            // Author's Choice Data
            authorsChoiceRankings: [],
            authorsChoiceLoading: false,

            // Run Benchmark Data
            benchmarkStatus: 'idle', // idle, running, completed, failed, cancelled
            benchmarkJob: null,
            benchmarkHistory: [],
            benchmarkLoading: false,
            benchmarkPolling: null,
            benchmarkLogs: [],
            benchmarkLibraryLogs: [],
            benchmarkAutoScroll: true,
            benchmarkLogPopout: null,
            showAdvancedLogs: false, // Toggle for showing library logs tab
            activeLogTab: 'benchmark', // 'benchmark' or 'library'
            benchmarkProgress: {
                current_model: null,
                current_model_index: 0,
                total_models: 0,
                current_phase: 'initializing',
                questions_completed: 0,
                questions_total: 0,
                models_completed: 0,
                elapsed_seconds: 0,
                cumulative_prompt_tokens: 0,
                cumulative_completion_tokens: 0,
                cumulative_reasoning_tokens: 0,
                cumulative_cost: 0.0
            },
            benchmarkConfig: {
                models: [],
                categories: [],
                question_ids: [],
                max_concurrent: 10,
                provider_concurrency: {}
            },
            benchmarkConfigValid: true,
            benchmarkConfigErrors: [],

            // Config Editor Data
            configContent: '',
            originalYamlText: '', // Full YAML text with comments (working copy)
            configErrors: [],
            configSaving: false,
            configBackups: [],
            configSection: 'judge', // Current config section: judge, fixer, models, filtering, performance, other
            showYamlCodeModal: false,
            isSyncing: false, // Flag to prevent infinite loops between visual and YAML sync
            configLoading: false, // Loading state for config
            configLoadError: null, // Error message if config load fails
            yamlSyncTimeout: null, // Debounce timer for YAML → Visual sync
            visualSyncTimeout: null, // Debounce timer for Visual → YAML sync
            isEditingYaml: false, // Flag to track if user is actively editing YAML
            yamlEditTimeout: null, // Timer to detect when user stops editing YAML

            // Visual Config Editor Data
            visualConfig: {
                models: [],
                model_display_names: {},
                model_configs: {},
                judge_model: '',
                fixer_model: '',
                categories: [],
                question_ids: [],
                max_concurrent: 10,
                provider_concurrency: {},
                retry_settings: {
                    max_retries_per_key: 5,
                    global_timeout: 180
                },
                evaluation: {
                    pass_threshold: 60,
                    code_timeout: 10
                },
                viewer: {
                    host: '0.0.0.0',
                    port: 8000
                },
                code_formatting_instructions: {
                    enabled: true,
                    instruction: ''
                },
                questions_dir: 'questions',
                results_dir: 'results'
            },

            // Judge & Fixer model configs (separate from test models)
            judgeModelConfig: {
                system_instruction: '',
                system_instruction_position: 'prepend',
                options: {}
            },
            fixerModelConfig: {
                system_instruction: '',
                system_instruction_position: 'prepend',
                options: {}
            },
            judgeModelOptionsJSON: '',
            fixerModelOptionsJSON: '',

            // Model editor UI state
            showAddModel: false,
            showAddProvider: false,
            modelSearchQuery: '',
            expandedModels: {},
            editingModelConfigs: {},
            editingModelOptionsJSON: {},
            newModel: {
                id: '',
                displayName: '',
                systemInstruction: '',
                position: 'prepend',
                optionsJSON: ''
            },
            newProvider: {
                name: '',
                limit: 1
            },

            // Categories
            availableCategories: [],
            questionIdsText: '',

            // JSON validation errors
            jsonErrors: {},

            // Job Logs Modal
            showJobLogsModal: false,
            selectedJobForLogs: null,
            jobLogsLoading: false,
            selectedJobLogTab: 'benchmark' // 'benchmark' or 'library'
        };
    },

    computed: {
        availableRuns() {
            // Group runs by model
            const runsByModel = {};
            this.runs.forEach(run => {
                if (!runsByModel[run.model]) {
                    runsByModel[run.model] = [];
                }
                runsByModel[run.model].push(run);
            });
            return runsByModel;
        },

        modelsFromRuns() {
            const models = new Set();
            this.runs.forEach(run => models.add(run.model));
            return Array.from(models).sort();
        },

        filteredModels() {
            // Filter models based on search query
            const query = this.modelSearchQuery.toLowerCase();
            if (!query) return this.visualConfig.models;
            return this.visualConfig.models.filter(model => {
                const displayName = this.visualConfig.model_display_names[model] || '';
                return model.toLowerCase().includes(query) ||
                       displayName.toLowerCase().includes(query);
            });
        }
    },

    watch: {
        currentPage(newPage) {
            this.onPageChange(newPage);
        },

        showConfigModal(show) {
            if (show) {
                this.loadConfig();
                this.loadCategoriesDetailed();
            }
        },

        showYamlCodeModal(show) {
            if (show) {
                // Initialize CodeMirror when YAML modal opens (now that it's visible)
                this.$nextTick(() => {
                    this.initializeCodeMirror();

                    // Refresh CodeMirror to fix layout issues (line numbers, etc.)
                    if (this.configEditor) {
                        this.configEditor.refresh();
                    }
                });
            } else {
                // Reset editing flag when closing modal
                this.isEditingYaml = false;
                if (this.yamlEditTimeout) {
                    clearTimeout(this.yamlEditTimeout);
                }
            }
        },

        visualConfig: {
            handler() {
                // Prevent infinite loops during config load/sync
                if (this.isSyncing) return;

                // Debounce Visual → YAML sync (prevent freezing when typing in visual fields)
                if (this.visualSyncTimeout) {
                    clearTimeout(this.visualSyncTimeout);
                }
                this.visualSyncTimeout = setTimeout(() => {
                    this.syncVisualToYaml();
                }, 300); // 300ms debounce - faster than YAML sync since visual is lighter
            },
            deep: true
        },

        judgeModelConfig: {
            handler() {
                // Prevent infinite loops during config load/sync
                if (this.isSyncing) return;

                // Debounce to prevent excessive syncing
                if (this.visualSyncTimeout) {
                    clearTimeout(this.visualSyncTimeout);
                }
                this.visualSyncTimeout = setTimeout(() => {
                    this.syncJudgeFixerToVisualConfig();
                }, 300);
            },
            deep: true
        },

        fixerModelConfig: {
            handler() {
                // Prevent infinite loops during config load/sync
                if (this.isSyncing) return;

                // Debounce to prevent excessive syncing
                if (this.visualSyncTimeout) {
                    clearTimeout(this.visualSyncTimeout);
                }
                this.visualSyncTimeout = setTimeout(() => {
                    this.syncJudgeFixerToVisualConfig();
                }, 300);
            },
            deep: true
        },

        questionIdsText(val) {
            // Parse comma-separated question IDs
            this.visualConfig.question_ids = val
                .split(',')
                .map(id => id.trim())
                .filter(id => id);
        }
    },

    mounted() {
        // Load initial data
        this.loadData();

        // Update icons when Vue renders
        this.$nextTick(() => {
            if (window.lucide) {
                lucide.createIcons();
            }
        });
    },

    updated() {
        // Re-initialize icons after any update
        this.$nextTick(() => {
            if (window.lucide) {
                lucide.createIcons();
            }
        });
    },

    methods: {
        // ====================================================================
        // General Data Loading
        // ====================================================================

        async loadData() {
            this.loading = true;
            try {
                await this.loadModelDisplayNames();
                await this.loadLeaderboard();
                await this.loadRuns();
                await this.loadQuestions();
                await this.loadUnifiedHumanJudgeLeaderboard();
            } catch (error) {
                console.error('Error loading data:', error);
                this.showToast('Failed to load data', 'error');
            } finally {
                this.loading = false;
            }
        },

        async loadLeaderboard() {
            try {
                const response = await fetch('/api/leaderboard/unified');
                const data = await response.json();
                this.leaderboard = data.leaderboard || [];
            } catch (error) {
                console.error('Error loading leaderboard:', error);
                this.leaderboard = [];
            }
        },

        async loadModelDisplayNames() {
            try {
                const response = await fetch('/api/model-display-names');
                const data = await response.json();
                this.modelDisplayNames = data.model_display_names || {};
            } catch (error) {
                console.error('Error loading display names:', error);
                this.modelDisplayNames = {};
            }
        },

        async loadRuns() {
            try {
                const response = await fetch('/api/runs');
                const data = await response.json();
                this.runs = data.runs || [];
            } catch (error) {
                console.error('Error loading runs:', error);
                this.runs = [];
            }
        },

        async loadQuestions() {
            try {
                const response = await fetch('/api/questions');
                const data = await response.json();
                this.questions = data.questions || [];
            } catch (error) {
                console.error('Error loading questions:', error);
                this.questions = [];
            }
        },

        async refreshData() {
            this.showToast('Refreshing data...', 'success');
            await this.loadData();
        },

        getDisplayName(modelName) {
            return this.modelDisplayNames[modelName] || modelName;
        },

        getScoreColor(score) {
            if (score >= 80) return 'text-green-400';
            if (score >= 60) return 'text-yellow-400';
            return 'text-red-400';
        },

        getScoreClass(score) {
            if (score >= 80) return 'score-excellent';
            if (score >= 60) return 'score-good';
            if (score >= 40) return 'score-fair';
            return 'score-poor';
        },

        // ====================================================================
        // Page Change Handler
        // ====================================================================

        async onPageChange(page) {
            switch (page) {
                case 'comparative':
                    await this.loadComparativeJobs();
                    break;
                case 'human':
                    await this.loadHumanJudgeData();
                    break;
                case 'authors-choice':
                    await this.loadAuthorsChoice();
                    break;
                case 'run-benchmark':
                    await this.loadBenchmarkPage();
                    break;
            }
        },

        // ====================================================================
        // Individual Evaluation Page
        // ====================================================================

        viewModelDetails(entry) {
            this.showToast('Model details view coming soon', 'success');
        },

        // ====================================================================
        // Comparative Judge Page
        // ====================================================================

        async loadComparativeJobs() {
            this.comparativeLoading = true;
            try {
                const response = await fetch('/api/comparative-judge/jobs');
                const data = await response.json();
                this.comparativeJobs = data.jobs || [];
            } catch (error) {
                console.error('Error loading comparative jobs:', error);
                this.comparativeJobs = [];
            } finally {
                this.comparativeLoading = false;
            }
        },

        openStartJobModal() {
            this.showStartJobModal = true;
            this.selectedRunsForComparison = [];
        },

        closeStartJobModal() {
            this.showStartJobModal = false;
        },

        toggleRunSelection(runId) {
            const index = this.selectedRunsForComparison.indexOf(runId);
            if (index > -1) {
                this.selectedRunsForComparison.splice(index, 1);
            } else {
                this.selectedRunsForComparison.push(runId);
            }
        },

        async startComparativeJob() {
            if (this.selectedRunsForComparison.length < 2) {
                this.showToast('Please select at least 2 runs to compare', 'error');
                return;
            }

            try {
                const response = await fetch('/api/comparative-judge/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        run_ids: this.selectedRunsForComparison
                    })
                });

                const data = await response.json();
                this.showToast('Comparative judging started', 'success');
                this.closeStartJobModal();
                await this.loadComparativeJobs();

                // Start polling for this job
                this.pollJobStatus(data.job_id);

            } catch (error) {
                console.error('Error starting job:', error);
                this.showToast('Failed to start comparative judging', 'error');
            }
        },

        async pollJobStatus(jobId) {
            const poll = async () => {
                try {
                    const response = await fetch(`/api/comparative-judge/jobs/${jobId}/status`);
                    const job = await response.json();

                    // Update job in list
                    const index = this.comparativeJobs.findIndex(j => j.job_id === jobId);
                    if (index > -1) {
                        this.$set(this.comparativeJobs, index, job);
                    }

                    // Continue polling if still running
                    if (job.status === 'running' || job.status === 'pending') {
                        setTimeout(poll, 2000);
                    } else if (job.status === 'completed') {
                        this.showToast('Comparative judging completed', 'success');
                    }
                } catch (error) {
                    console.error('Error polling job status:', error);
                }
            };

            poll();
        },

        async viewJobResults(jobId) {
            try {
                const response = await fetch(`/api/comparative-judge/jobs/${jobId}/results`);
                const data = await response.json();
                this.selectedJobResults = data;
            } catch (error) {
                console.error('Error loading job results:', error);
                this.showToast('Failed to load results', 'error');
            }
        },

        closeJobResults() {
            this.selectedJobResults = null;
        },

        // ====================================================================
        // Human Judge Page
        // ====================================================================

        async loadHumanJudgeData() {
            this.humanJudgeLoading = true;
            try {
                if (this.runs.length > 0 && !this.humanJudgeSelectedRun) {
                    this.humanJudgeSelectedRun = this.runs[0].run_id;
                }

                if (this.humanJudgeSelectedRun) {
                    await this.loadHumanJudgeLeaderboard();
                    await this.loadHumanJudgeQuestions();
                }
            } catch (error) {
                console.error('Error loading human judge data:', error);
            } finally {
                this.humanJudgeLoading = false;
            }
        },

        async loadUnifiedHumanJudgeLeaderboard() {
            this.humanJudgeUnifiedLoading = true;
            try {
                const response = await fetch('/api/human-ratings/unified-leaderboard');
                const data = await response.json();
                this.humanJudgeUnifiedLeaderboard = data.leaderboard || [];
            } catch (error) {
                console.error('Error loading unified human judge leaderboard:', error);
                this.humanJudgeUnifiedLeaderboard = [];
            } finally {
                this.humanJudgeUnifiedLoading = false;
            }
        },

        async loadHumanJudgeLeaderboard() {
            if (!this.humanJudgeSelectedRun) return;

            try {
                const response = await fetch(`/api/runs/${this.humanJudgeSelectedRun}/human-ratings/leaderboard`);
                const data = await response.json();
                this.humanJudgeLeaderboard = data.leaderboard || [];
            } catch (error) {
                console.error('Error loading human judge leaderboard:', error);
                this.humanJudgeLeaderboard = [];
            }
        },

        async loadHumanJudgeQuestions() {
            if (!this.humanJudgeSelectedRun) return;

            try {
                const run = this.runs.find(r => r.run_id === this.humanJudgeSelectedRun);
                if (!run) return;

                // Get all responses for this run
                const response = await fetch(`/api/runs/${this.humanJudgeSelectedRun}/bulk-data?model_name=${run.model}`);
                const data = await response.json();

                this.humanJudgeQuestions = Object.keys(data.responses || {}).map(qid => ({
                    id: qid,
                    ...data.questions[qid],
                    response: data.responses[qid],
                    rating: null
                }));

                // Load existing ratings
                for (const q of this.humanJudgeQuestions) {
                    const ratingResponse = await fetch(`/api/runs/${this.humanJudgeSelectedRun}/human-ratings/${run.model}/${q.id}`);
                    const ratingData = await ratingResponse.json();
                    q.rating = ratingData.rating;
                }

            } catch (error) {
                console.error('Error loading human judge questions:', error);
                this.humanJudgeQuestions = [];
            }
        },

        openRatingModal(question) {
            this.selectedQuestionForRating = question;
            this.currentRating = {
                score: question.rating?.score || 50,
                comment: question.rating?.comment || ''
            };
        },

        closeRatingModal() {
            this.selectedQuestionForRating = null;
        },

        async saveRating() {
            if (!this.selectedQuestionForRating) return;

            try {
                const run = this.runs.find(r => r.run_id === this.humanJudgeSelectedRun);
                if (!run) return;

                const response = await fetch(
                    `/api/runs/${this.humanJudgeSelectedRun}/human-ratings/${run.model}/${this.selectedQuestionForRating.id}`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(this.currentRating)
                    }
                );

                const data = await response.json();
                this.showToast('Rating saved', 'success');
                this.closeRatingModal();

                // Reload data
                await this.loadUnifiedHumanJudgeLeaderboard();
                await this.loadHumanJudgeLeaderboard();
                await this.loadHumanJudgeQuestions();

            } catch (error) {
                console.error('Error saving rating:', error);
                this.showToast('Failed to save rating', 'error');
            }
        },

        // ====================================================================
        // Author's Choice Page
        // ====================================================================

        async loadAuthorsChoice() {
            this.authorsChoiceLoading = true;
            try {
                const response = await fetch('/api/authors-choice');
                const data = await response.json();

                // If no rankings exist, create default from models
                if (!data.rankings || data.rankings.length === 0) {
                    this.authorsChoiceRankings = this.modelsFromRuns.map((model, index) => ({
                        model_name: model,
                        position: index + 1
                    }));
                } else {
                    this.authorsChoiceRankings = data.rankings;
                }

                // Initialize sortable
                this.$nextTick(() => {
                    this.initializeSortable();
                });

            } catch (error) {
                console.error('Error loading author\'s choice:', error);
                this.authorsChoiceRankings = [];
            } finally {
                this.authorsChoiceLoading = false;
            }
        },

        initializeSortable() {
            const el = document.getElementById('sortable-rankings');
            if (!el || !window.Sortable) return;

            Sortable.create(el, {
                animation: 150,
                handle: '.drag-handle',
                onEnd: () => {
                    // Update positions after drag
                    const items = Array.from(el.children);
                    this.authorsChoiceRankings = items.map((item, index) => ({
                        model_name: item.dataset.modelName,
                        position: index + 1
                    }));
                }
            });
        },

        async saveAuthorsChoice() {
            this.authorsChoiceLoading = true;
            try {
                const response = await fetch('/api/authors-choice', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        rankings: this.authorsChoiceRankings
                    })
                });

                const data = await response.json();
                this.showToast('Rankings saved', 'success');

            } catch (error) {
                console.error('Error saving rankings:', error);
                this.showToast('Failed to save rankings', 'error');
            } finally {
                this.authorsChoiceLoading = false;
            }
        },

        resetAuthorsChoice() {
            this.authorsChoiceRankings = this.modelsFromRuns.map((model, index) => ({
                model_name: model,
                position: index + 1
            }));
        },

        // ====================================================================
        // Config Editor Modal
        // ====================================================================

        async loadConfig() {
            this.configLoading = true;
            this.configLoadError = null;

            // Prevent watchers from triggering during load (avoid infinite loops)
            this.isSyncing = true;

            // Timeout after 10 seconds
            const timeout = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Config load timeout (10s)')), 10000);
            });

            try {
                // Race between fetch and timeout
                const fetchPromise = fetch('/api/config').then(r => {
                    if (!r.ok) throw new Error(`API error: ${r.status}`);
                    return r.json();
                });
                const data = await Promise.race([fetchPromise, timeout]);

                // Store complete YAML with comments (working copy)
                this.originalYamlText = data.content || '';
                this.configContent = this.originalYamlText;

                // Parse YAML for visual editor
                try {
                    const parsed = jsyaml.load(this.originalYamlText);
                    this.visualConfig = {
                        models: parsed.models || [],
                        model_display_names: parsed.model_display_names || {},
                        model_configs: parsed.model_configs || {},
                        judge_model: parsed.judge_model || '',
                        fixer_model: parsed.fixer_model || '',
                        categories: parsed.categories || [],
                        question_ids: parsed.question_ids || [],
                        max_concurrent: parsed.max_concurrent || 10,
                        provider_concurrency: parsed.provider_concurrency || {},
                        retry_settings: parsed.retry_settings || { max_retries_per_key: 5, global_timeout: 180 },
                        evaluation: parsed.evaluation || { pass_threshold: 60, code_timeout: 10 },
                        viewer: parsed.viewer || { host: '0.0.0.0', port: 8000 },
                        code_formatting_instructions: parsed.code_formatting_instructions || { enabled: true, instruction: '' },
                        questions_dir: parsed.questions_dir || 'questions',
                        results_dir: parsed.results_dir || 'results'
                    };

                    // Populate judge and fixer configs from model_configs
                    if (this.visualConfig.judge_model && this.visualConfig.model_configs[this.visualConfig.judge_model]) {
                        const jc = this.visualConfig.model_configs[this.visualConfig.judge_model];
                        this.judgeModelConfig = {
                            system_instruction: jc.system_instruction || '',
                            system_instruction_position: jc.system_instruction_position || 'prepend',
                            options: jc.options || {}
                        };
                        // Strip outer braces for display
                        this.judgeModelOptionsJSON = jc.options ? this.stripOuterBraces(JSON.stringify(jc.options, null, 2)) : '';
                    }

                    if (this.visualConfig.fixer_model && this.visualConfig.model_configs[this.visualConfig.fixer_model]) {
                        const fc = this.visualConfig.model_configs[this.visualConfig.fixer_model];
                        this.fixerModelConfig = {
                            system_instruction: fc.system_instruction || '',
                            system_instruction_position: fc.system_instruction_position || 'prepend',
                            options: fc.options || {}
                        };
                        // Strip outer braces for display
                        this.fixerModelOptionsJSON = fc.options ? this.stripOuterBraces(JSON.stringify(fc.options, null, 2)) : '';
                    }

                    // Convert question_ids array to text
                    this.questionIdsText = this.visualConfig.question_ids.join(', ');

                    // Initialize editing configs for all models
                    this.visualConfig.models.forEach(model => {
                        const config = this.visualConfig.model_configs[model] || {};
                        this.editingModelConfigs[model] = {
                            system_instruction: config.system_instruction || '',
                            system_instruction_position: config.system_instruction_position || 'prepend',
                            options: config.options || {}
                        };
                        // Strip outer braces for display
                        this.editingModelOptionsJSON[model] = config.options ? this.stripOuterBraces(JSON.stringify(config.options, null, 2)) : '';
                    });

                } catch (yamlError) {
                    console.error('Error parsing YAML:', yamlError);
                    this.configLoadError = 'Error parsing config YAML: ' + yamlError.message;
                    this.showToast('Error parsing config YAML', 'error');
                    this.configLoading = false;
                    this.isSyncing = false; // Re-enable watchers on YAML parse error
                    return;
                }

                // Load backups
                await this.loadConfigBackups();

                // Wait for DOM to update
                await this.$nextTick();

                // Mark loading complete
                this.configLoading = false;

                // Re-enable watchers now that initial load is done
                this.isSyncing = false;

            } catch (error) {
                console.error('Error loading config:', error);
                this.configLoadError = error.message || 'Failed to load config';
                this.showToast('Failed to load config: ' + (error.message || 'Unknown error'), 'error');
                this.configLoading = false;
                this.isSyncing = false; // Re-enable watchers even on error

                // Re-render icons for error state
                this.$nextTick(() => {
                    if (window.lucide) {
                        lucide.createIcons();
                    }
                });
            }
        },

        initializeCodeMirror() {
            const textarea = document.getElementById('config-editor');
            if (!textarea || !window.CodeMirror) return;

            // Check if already initialized
            if (this.configEditor) {
                // Already initialized, just update content and refresh
                try {
                    this.isSyncing = true;
                    this.configEditor.setValue(this.originalYamlText);
                    this.configEditor.refresh();
                    this.isSyncing = false;
                } catch (error) {
                    console.error('Error updating CodeMirror:', error);
                    this.isSyncing = false;
                }
                return;
            }

            // Not initialized yet, create new instance
            try {
                const editor = CodeMirror.fromTextArea(textarea, {
                    mode: 'yaml',
                    theme: 'dracula',
                    lineNumbers: true,
                    indentUnit: 2,
                    tabSize: 2,
                    lineWrapping: true,
                    viewportMargin: Infinity,
                });

                // Set editor height
                editor.setSize(null, '500px');

                // Handle YAML changes and sync to visual editor
                editor.on('change', (cm) => {
                    // Prevent infinite loops
                    if (this.isSyncing) return;

                    const yamlText = cm.getValue();
                    this.configContent = yamlText;
                    this.originalYamlText = yamlText;

                    // Mark that user is actively editing YAML
                    this.isEditingYaml = true;

                    // Clear previous timers
                    if (this.yamlSyncTimeout) {
                        clearTimeout(this.yamlSyncTimeout);
                    }
                    if (this.yamlEditTimeout) {
                        clearTimeout(this.yamlEditTimeout);
                    }

                    // Sync YAML → Visual after user stops typing
                    this.yamlSyncTimeout = setTimeout(() => {
                        this.syncYamlToVisual(yamlText);
                    }, 500);

                    // Mark editing as finished 2 seconds after last keystroke
                    this.yamlEditTimeout = setTimeout(() => {
                        this.isEditingYaml = false;
                    }, 2000);
                });

                // Store editor instance
                this.configEditor = editor;

                // Set initial content
                this.isSyncing = true;
                editor.setValue(this.originalYamlText);
                this.isSyncing = false;

                // Refresh to ensure proper rendering
                editor.refresh();
            } catch (error) {
                console.error('Error initializing CodeMirror:', error);
            }
        },

        async validateConfig() {
            try {
                const response = await fetch('/api/config/validate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        yaml_content: this.configContent
                    })
                });

                const data = await response.json();

                if (data.valid) {
                    this.configErrors = [];
                    this.showToast('Config is valid', 'success');
                } else {
                    this.configErrors = data.errors || [];
                    this.showToast('Config validation failed', 'error');
                }

            } catch (error) {
                console.error('Error validating config:', error);
                this.showToast('Validation error', 'error');
            }
        },

        async saveConfig() {
            // Show saving immediately (non-blocking UI)
            this.configSaving = true;

            // Validate first (in background)
            await this.validateConfig();

            if (this.configErrors.length > 0) {
                this.configSaving = false;
                this.showToast('Please fix validation errors before saving', 'error');
                return;
            }

            try {
                // Save in background without blocking
                const savePromise = fetch('/api/config/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        // Save originalYamlText (working copy with comments when possible)
                        yaml_content: this.originalYamlText
                    })
                }).then(r => r.json());

                // Handle response asynchronously
                savePromise.then(data => {
                    this.showToast(data.message || 'Config saved', 'success');
                    this.configSaving = false;

                    // Reload backups in background
                    this.loadConfigBackups();
                }).catch(error => {
                    console.error('Error saving config:', error);
                    this.showToast('Failed to save config', 'error');
                    this.configSaving = false;
                });

            } catch (error) {
                console.error('Error saving config:', error);
                this.showToast('Failed to save config', 'error');
                this.configSaving = false;
            }
        },

        async loadConfigBackups() {
            try {
                const response = await fetch('/api/config/backups');
                const data = await response.json();
                this.configBackups = data.backups || [];
            } catch (error) {
                console.error('Error loading backups:', error);
                this.configBackups = [];
            }
        },

        async restoreConfigBackup(backupName) {
            if (!confirm(`Restore config from ${backupName}?`)) return;

            try {
                const response = await fetch(`/api/config/restore/${backupName}`, {
                    method: 'POST'
                });

                const data = await response.json();
                this.showToast(data.message, 'success');

                // Reload config
                await this.loadConfig();

            } catch (error) {
                console.error('Error restoring backup:', error);
                this.showToast('Failed to restore backup', 'error');
            }
        },

        async deleteConfigBackup(backupName) {
            if (!confirm(`Delete backup ${backupName}? This cannot be undone.`)) return;

            try {
                const response = await fetch(`/api/config/backups/${backupName}`, {
                    method: 'DELETE'
                });

                const data = await response.json();
                this.showToast(data.message, 'success');

                // Reload backups list
                await this.loadConfigBackups();

            } catch (error) {
                console.error('Error deleting backup:', error);
                this.showToast('Failed to delete backup', 'error');
            }
        },

        // ====================================================================
        // Visual Config Editor Methods
        // ====================================================================

        syncVisualToYaml() {
            // Don't update YAML editor if user is actively editing it
            if (this.isEditingYaml) {
                return;
            }

            // Sync judge and fixer configs into model_configs
            this.syncJudgeFixerToVisualConfig();

            // Convert visualConfig to YAML
            try {
                const yamlContent = jsyaml.dump(this.visualConfig, {
                    indent: 2,
                    lineWidth: 120,
                    noRefs: true,
                    sortKeys: false,
                    quotingType: '"',  // Use double quotes for strings (preserves original style)
                    forceQuotes: true  // Force quotes on all strings
                });

                // Update originalYamlText (best-effort: regenerate YAML)
                // Note: Comments from original YAML will be lost when visual editor changes
                // This is acceptable per user's "best effort" requirement
                this.originalYamlText = yamlContent;
                this.configContent = yamlContent;

                // Update CodeMirror if it exists and content has changed
                if (this.configEditor) {
                    const currentValue = this.configEditor.getValue();

                    // Only update if content actually changed (prevents unnecessary updates)
                    if (currentValue !== yamlContent) {
                        this.isSyncing = true;
                        this.configEditor.setValue(yamlContent);
                        this.isSyncing = false;
                    }
                }
            } catch (error) {
                console.error('Error converting to YAML:', error);
                this.isSyncing = false;
            }
        },

        syncJudgeFixerToVisualConfig() {
            // Sync judge model config
            if (this.visualConfig.judge_model) {
                const judgeConfig = {};
                if (this.judgeModelConfig.system_instruction) {
                    judgeConfig.system_instruction = this.judgeModelConfig.system_instruction;
                    judgeConfig.system_instruction_position = this.judgeModelConfig.system_instruction_position;
                }
                if (Object.keys(this.judgeModelConfig.options).length > 0) {
                    judgeConfig.options = this.judgeModelConfig.options;
                }
                if (Object.keys(judgeConfig).length > 0) {
                    this.visualConfig.model_configs[this.visualConfig.judge_model] = judgeConfig;
                }
            }

            // Sync fixer model config
            if (this.visualConfig.fixer_model) {
                const fixerConfig = {};
                if (this.fixerModelConfig.system_instruction) {
                    fixerConfig.system_instruction = this.fixerModelConfig.system_instruction;
                    fixerConfig.system_instruction_position = this.fixerModelConfig.system_instruction_position;
                }
                if (Object.keys(this.fixerModelConfig.options).length > 0) {
                    fixerConfig.options = this.fixerModelConfig.options;
                }
                if (Object.keys(fixerConfig).length > 0) {
                    this.visualConfig.model_configs[this.visualConfig.fixer_model] = fixerConfig;
                }
            }
        },

        syncYamlToVisual(yamlText) {
            // Parse YAML and update visual editor fields (YAML → Visual sync)
            try {
                const parsed = jsyaml.load(yamlText);
                if (!parsed) return;

                // Update visualConfig
                this.visualConfig.models = parsed.models || [];
                this.visualConfig.model_display_names = parsed.model_display_names || {};
                this.visualConfig.model_configs = parsed.model_configs || {};
                this.visualConfig.judge_model = parsed.judge_model || '';
                this.visualConfig.fixer_model = parsed.fixer_model || '';
                this.visualConfig.categories = parsed.categories || [];
                this.visualConfig.question_ids = parsed.question_ids || [];
                this.visualConfig.max_concurrent = parsed.max_concurrent || 10;
                this.visualConfig.provider_concurrency = parsed.provider_concurrency || {};
                this.visualConfig.retry_settings = parsed.retry_settings || { max_retries_per_key: 5, global_timeout: 180 };
                this.visualConfig.evaluation = parsed.evaluation || { pass_threshold: 60, code_timeout: 10 };
                this.visualConfig.viewer = parsed.viewer || { host: '0.0.0.0', port: 8000 };
                this.visualConfig.code_formatting_instructions = parsed.code_formatting_instructions || { enabled: true, instruction: '' };
                this.visualConfig.questions_dir = parsed.questions_dir || 'questions';
                this.visualConfig.results_dir = parsed.results_dir || 'results';

                // Update judge and fixer configs
                if (this.visualConfig.judge_model && this.visualConfig.model_configs[this.visualConfig.judge_model]) {
                    const jc = this.visualConfig.model_configs[this.visualConfig.judge_model];
                    this.judgeModelConfig.system_instruction = jc.system_instruction || '';
                    this.judgeModelConfig.system_instruction_position = jc.system_instruction_position || 'prepend';
                    this.judgeModelConfig.options = jc.options || {};
                    this.judgeModelOptionsJSON = jc.options ? this.stripOuterBraces(JSON.stringify(jc.options, null, 2)) : '';
                }

                if (this.visualConfig.fixer_model && this.visualConfig.model_configs[this.visualConfig.fixer_model]) {
                    const fc = this.visualConfig.model_configs[this.visualConfig.fixer_model];
                    this.fixerModelConfig.system_instruction = fc.system_instruction || '';
                    this.fixerModelConfig.system_instruction_position = fc.system_instruction_position || 'prepend';
                    this.fixerModelConfig.options = fc.options || {};
                    this.fixerModelOptionsJSON = fc.options ? this.stripOuterBraces(JSON.stringify(fc.options, null, 2)) : '';
                }

                // Update question IDs text
                this.questionIdsText = this.visualConfig.question_ids.join(', ');

                // Update editing configs for models
                this.visualConfig.models.forEach(model => {
                    const config = this.visualConfig.model_configs[model] || {};
                    this.editingModelConfigs[model] = {
                        system_instruction: config.system_instruction || '',
                        system_instruction_position: config.system_instruction_position || 'prepend',
                        options: config.options || {}
                    };
                    this.editingModelOptionsJSON[model] = config.options ? this.stripOuterBraces(JSON.stringify(config.options, null, 2)) : '';
                });

            } catch (error) {
                // Silent fail - don't show errors during typing
                console.error('YAML parse error (during sync):', error);
            }
        },

        validateJSON(type) {
            // Validate JSON or YAML for judge or fixer (accepts both formats)
            let inputText = type === 'judge' ? this.judgeModelOptionsJSON : this.fixerModelOptionsJSON;

            if (!inputText.trim()) {
                this.jsonErrors[type] = '';
                if (type === 'judge') {
                    this.judgeModelConfig.options = {};
                } else {
                    this.fixerModelConfig.options = {};
                }
                return;
            }

            let parsed = null;
            inputText = inputText.trim();

            // Try JSON first (with auto-added braces)
            let jsonText = inputText.startsWith('{') ? inputText : '{' + inputText + '}';
            try {
                parsed = JSON.parse(jsonText);
                this.jsonErrors[type] = '';
            } catch (jsonError) {
                // JSON failed, try YAML (supports both "field": "value" and field: value)
                try {
                    parsed = jsyaml.load(jsonText);
                    this.jsonErrors[type] = '';
                } catch (yamlError) {
                    this.jsonErrors[type] = 'Invalid JSON/YAML: ' + jsonError.message;
                    return;
                }
            }

            if (type === 'judge') {
                this.judgeModelConfig.options = parsed;
            } else {
                this.fixerModelConfig.options = parsed;
            }
        },

        validateModelOptionsJSON(modelId) {
            // Validate JSON or YAML for model options (accepts both formats)
            let inputText = this.editingModelOptionsJSON[modelId];

            if (!inputText || !inputText.trim()) {
                this.jsonErrors['model_' + modelId] = '';
                this.editingModelConfigs[modelId].options = {};
                return;
            }

            let parsed = null;
            inputText = inputText.trim();

            // Try JSON first (with auto-added braces)
            let jsonText = inputText.startsWith('{') ? inputText : '{' + inputText + '}';
            try {
                parsed = JSON.parse(jsonText);
                this.jsonErrors['model_' + modelId] = '';
            } catch (jsonError) {
                // JSON failed, try YAML (supports both "field": "value" and field: value)
                try {
                    parsed = jsyaml.load(jsonText);
                    this.jsonErrors['model_' + modelId] = '';
                } catch (yamlError) {
                    this.jsonErrors['model_' + modelId] = 'Invalid JSON/YAML: ' + jsonError.message;
                    return;
                }
            }

            this.editingModelConfigs[modelId].options = parsed;
        },

        addModel() {
            if (!this.newModel.id) return;

            // Add to models list
            if (!this.visualConfig.models.includes(this.newModel.id)) {
                this.visualConfig.models.push(this.newModel.id);
            }

            // Add display name if provided
            if (this.newModel.displayName) {
                this.visualConfig.model_display_names[this.newModel.id] = this.newModel.displayName;
            }

            // Add model config if advanced settings provided
            if (this.newModel.systemInstruction || this.newModel.optionsJSON) {
                const config = {};
                if (this.newModel.systemInstruction) {
                    config.system_instruction = this.newModel.systemInstruction;
                    config.system_instruction_position = this.newModel.position;
                }
                if (this.newModel.optionsJSON) {
                    let inputText = this.newModel.optionsJSON.trim();
                    let jsonText = inputText.startsWith('{') ? inputText : '{' + inputText + '}';

                    try {
                        // Try JSON first
                        config.options = JSON.parse(jsonText);
                    } catch (jsonError) {
                        // JSON failed, try YAML (supports both "field": "value" and field: value)
                        try {
                            config.options = jsyaml.load(jsonText);
                        } catch (yamlError) {
                            this.showToast('Invalid JSON/YAML in options', 'error');
                            return;
                        }
                    }
                }
                this.visualConfig.model_configs[this.newModel.id] = config;

                // Initialize editing config
                this.editingModelConfigs[this.newModel.id] = { ...config };
                // Strip outer braces for display
                this.editingModelOptionsJSON[this.newModel.id] = config.options ?
                    this.stripOuterBraces(JSON.stringify(config.options, null, 2)) : '';
            }

            this.showToast(`Model ${this.newModel.id} added`, 'success');
            this.cancelAddModel();
        },

        cancelAddModel() {
            this.showAddModel = false;
            this.newModel = {
                id: '',
                displayName: '',
                systemInstruction: '',
                position: 'prepend',
                optionsJSON: ''
            };
        },

        deleteModel(modelId) {
            if (!confirm(`Delete model ${modelId}?`)) return;

            // Remove from models list
            const index = this.visualConfig.models.indexOf(modelId);
            if (index > -1) {
                this.visualConfig.models.splice(index, 1);
            }

            // Remove from display names and configs
            delete this.visualConfig.model_display_names[modelId];
            delete this.visualConfig.model_configs[modelId];
            delete this.editingModelConfigs[modelId];
            delete this.editingModelOptionsJSON[modelId];
            delete this.expandedModels[modelId];

            this.showToast(`Model ${modelId} deleted`, 'success');
        },

        toggleModelExpand(modelId) {
            this.expandedModels[modelId] = !this.expandedModels[modelId];

            // Initialize editing config if not already
            if (this.expandedModels[modelId] && !this.editingModelConfigs[modelId]) {
                const config = this.visualConfig.model_configs[modelId] || {};
                this.editingModelConfigs[modelId] = {
                    system_instruction: config.system_instruction || '',
                    system_instruction_position: config.system_instruction_position || 'prepend',
                    options: config.options || {}
                };
                this.editingModelOptionsJSON[modelId] = config.options ?
                    this.stripOuterBraces(JSON.stringify(config.options, null, 2)) : '';
            }

            // Re-render icons after DOM update
            this.$nextTick(() => {
                if (window.lucide) {
                    lucide.createIcons();
                }
            });
        },

        expandModel(modelId) {
            // Only expand if not already expanded (don't collapse)
            if (!this.expandedModels[modelId]) {
                this.expandedModels[modelId] = true;

                // Initialize editing config if not already
                if (!this.editingModelConfigs[modelId]) {
                    const config = this.visualConfig.model_configs[modelId] || {};
                    this.editingModelConfigs[modelId] = {
                        system_instruction: config.system_instruction || '',
                        system_instruction_position: config.system_instruction_position || 'prepend',
                        options: config.options || {}
                    };
                    this.editingModelOptionsJSON[modelId] = config.options ?
                        this.stripOuterBraces(JSON.stringify(config.options, null, 2)) : '';
                }

                // Re-render icons
                this.$nextTick(() => {
                    if (window.lucide) {
                        lucide.createIcons();
                    }
                });
            }
        },

        saveModelConfig(modelId) {
            // Validate JSON first
            this.validateModelOptionsJSON(modelId);
            if (this.jsonErrors['model_' + modelId]) {
                this.showToast('Fix JSON errors before saving', 'error');
                return;
            }

            // Prevent watchers from triggering during save
            this.isSyncing = true;

            // Save config
            const config = this.editingModelConfigs[modelId];
            if (config.system_instruction || Object.keys(config.options).length > 0) {
                this.visualConfig.model_configs[modelId] = { ...config };
            } else {
                delete this.visualConfig.model_configs[modelId];
            }

            // Re-enable watchers
            this.isSyncing = false;

            this.showToast(`Model ${modelId} configuration saved`, 'success');

            // Collapse the model card (don't use toggle to avoid triggering watchers)
            this.expandedModels[modelId] = false;
        },

        addProviderLimit() {
            if (!this.newProvider.name || !this.newProvider.limit) return;

            this.visualConfig.provider_concurrency[this.newProvider.name] = this.newProvider.limit;

            this.showAddProvider = false;
            this.newProvider = { name: '', limit: 1 };
            this.showToast('Provider limit added', 'success');
        },

        deleteProviderLimit(provider) {
            delete this.visualConfig.provider_concurrency[provider];
            this.showToast(`Provider ${provider} limit removed`, 'success');
        },

        stripOuterBraces(jsonString) {
            // Strip outer { } from JSON string for display
            if (!jsonString) return '';
            const trimmed = jsonString.trim();
            if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
                // Remove first { and last }, keeping inner content
                const inner = trimmed.slice(1, -1).trim();
                return inner;
            }
            return jsonString;
        },

        async loadCategoriesDetailed() {
            try {
                const response = await fetch('/api/categories/detailed');
                const data = await response.json();
                this.availableCategories = data.categories || [];
            } catch (error) {
                console.error('Error loading categories:', error);
                // Fallback to basic categories endpoint
                try {
                    const response = await fetch('/api/categories');
                    const data = await response.json();
                    this.availableCategories = (data.categories || []).map(name => ({
                        name,
                        display_name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                        question_count: 0
                    }));
                } catch (fallbackError) {
                    console.error('Error loading categories (fallback):', fallbackError);
                    this.availableCategories = [];
                }
            }
        },

        // ====================================================================
        // Run Benchmark Page
        // ====================================================================

        async loadBenchmarkPage() {
            try {
                // Load configuration to populate benchmark config
                await this.loadBenchmarkConfig();

                // Load job history
                await this.loadBenchmarkHistory();

                // Check current status
                await this.pollBenchmarkStatus();

                // Start polling if running
                if (this.benchmarkStatus === 'running') {
                    this.startBenchmarkPolling();
                }
            } catch (error) {
                console.error('Error loading benchmark page:', error);
                this.showToast('Failed to load benchmark page', 'error');
            }
        },

        async loadBenchmarkConfig() {
            try {
                // Load config from server
                const response = await fetch('/api/config');
                const data = await response.json();

                // Parse YAML
                const config = jsyaml.load(data.content);

                // Populate benchmark config from loaded config
                this.benchmarkConfig = {
                    models: config.models || [],
                    categories: config.categories || [],
                    question_ids: config.question_ids || [],
                    max_concurrent: config.max_concurrent || 10,
                    provider_concurrency: config.provider_concurrency || {}
                };

                // Validate config
                this.validateBenchmarkConfig();
            } catch (error) {
                console.error('Error loading benchmark config:', error);
                this.benchmarkConfigValid = false;
                this.benchmarkConfigErrors = ['Failed to load configuration'];
            }
        },

        validateBenchmarkConfig() {
            const errors = [];

            if (!this.benchmarkConfig.models || this.benchmarkConfig.models.length === 0) {
                errors.push('At least one model must be configured');
            }

            this.benchmarkConfigErrors = errors;
            this.benchmarkConfigValid = errors.length === 0;
        },

        async startBenchmark() {
            try {
                this.benchmarkLoading = true;

                const response = await fetch('/api/benchmark/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.benchmarkConfig)
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to start benchmark');
                }

                const data = await response.json();
                this.showToast('Benchmark started successfully', 'success');

                // Start polling for status
                this.startBenchmarkPolling();

            } catch (error) {
                console.error('Error starting benchmark:', error);
                this.showToast(error.message, 'error');
            } finally {
                this.benchmarkLoading = false;
            }
        },

        async stopBenchmark() {
            try {
                this.benchmarkLoading = true;

                const response = await fetch('/api/benchmark/stop', {
                    method: 'DELETE'
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to stop benchmark');
                }

                this.showToast('Benchmark stop requested', 'warning');

                // Immediately poll to get updated status
                await this.pollBenchmarkStatus();

            } catch (error) {
                console.error('Error stopping benchmark:', error);
                this.showToast(error.message, 'error');
            } finally {
                this.benchmarkLoading = false;
            }
        },

        async pollBenchmarkStatus() {
            try {
                const response = await fetch('/api/benchmark/status');
                const data = await response.json();

                this.benchmarkStatus = data.status;

                if (data.job) {
                    this.benchmarkJob = data.job;
                    this.benchmarkProgress = data.job.progress;
                    this.benchmarkLogs = data.job.logs || [];
                    this.benchmarkLibraryLogs = data.job.library_logs || [];

                    // Auto-scroll logs to bottom (if enabled)
                    if (this.benchmarkAutoScroll) {
                        this.$nextTick(() => {
                            // Auto-scroll the active tab
                            const logContainer = this.activeLogTab === 'benchmark'
                                ? document.querySelector('.benchmark-logs-container')
                                : document.querySelector('.library-logs-container');
                            if (logContainer) {
                                logContainer.scrollTop = logContainer.scrollHeight;
                            }
                        });
                    }

                    // Update pop-out window if open
                    if (this.benchmarkLogPopout && !this.benchmarkLogPopout.closed) {
                        this.updatePopoutWindow();
                    }
                }

                // If completed or failed, stop polling
                if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
                    this.stopBenchmarkPolling();

                    // Reload history
                    await this.loadBenchmarkHistory();

                    // Show notification
                    if (data.status === 'completed') {
                        this.showToast('Benchmark completed successfully!', 'success');
                    } else if (data.status === 'failed') {
                        this.showToast('Benchmark failed: ' + (data.job?.error || 'Unknown error'), 'error');
                    } else if (data.status === 'cancelled') {
                        this.showToast('Benchmark cancelled', 'warning');
                    }
                }

            } catch (error) {
                console.error('Error polling benchmark status:', error);
            }
        },

        startBenchmarkPolling() {
            // Stop any existing polling
            this.stopBenchmarkPolling();

            // Poll every 1 second
            this.benchmarkPolling = setInterval(() => {
                this.pollBenchmarkStatus();
            }, 1000);
        },

        stopBenchmarkPolling() {
            if (this.benchmarkPolling) {
                clearInterval(this.benchmarkPolling);
                this.benchmarkPolling = null;
            }
        },

        async loadBenchmarkHistory() {
            try {
                const response = await fetch('/api/benchmark/history?limit=10');
                const data = await response.json();

                this.benchmarkHistory = data.history || [];
            } catch (error) {
                console.error('Error loading benchmark history:', error);
            }
        },

        toggleAutoScroll() {
            this.benchmarkAutoScroll = !this.benchmarkAutoScroll;

            // If enabling, scroll to bottom immediately
            if (this.benchmarkAutoScroll) {
                this.$nextTick(() => {
                    const logContainer = document.querySelector('.benchmark-logs-container');
                    if (logContainer) {
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }
                });
            }
        },

        openLogPopout() {
            // Close existing popout if any
            if (this.benchmarkLogPopout && !this.benchmarkLogPopout.closed) {
                this.benchmarkLogPopout.close();
            }

            // Open new popout window
            const width = 800;
            const height = 600;
            const left = (screen.width - width) / 2;
            const top = (screen.height - height) / 2;

            this.benchmarkLogPopout = window.open(
                '/benchmark-log-popout',
                'BenchmarkLogPopout',
                `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`
            );

            // Wait for window to load, then send initial data
            if (this.benchmarkLogPopout) {
                setTimeout(() => {
                    this.updatePopoutWindow();
                }, 100);
            }
        },

        updatePopoutWindow() {
            if (!this.benchmarkLogPopout || this.benchmarkLogPopout.closed) {
                return;
            }

            try {
                // Convert Vue reactive objects to plain objects for postMessage
                // This is necessary because Vue 3 Proxy objects cannot be cloned
                const plainData = JSON.parse(JSON.stringify({
                    type: 'benchmark-update',
                    status: this.benchmarkStatus,
                    job: this.benchmarkJob,
                    progress: this.benchmarkProgress,
                    logs: this.benchmarkLogs,
                    libraryLogs: this.benchmarkLibraryLogs
                }));

                // Send data to popout window
                this.benchmarkLogPopout.postMessage(plainData, window.location.origin);
            } catch (error) {
                console.error('Error updating popout window:', error);
            }
        },

        formatDuration(seconds) {
            if (!seconds) return '0s';

            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);

            if (hours > 0) {
                return `${hours}h ${minutes}m ${secs}s`;
            } else if (minutes > 0) {
                return `${minutes}m ${secs}s`;
            } else {
                return `${secs}s`;
            }
        },

        formatTimestamp(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            return date.toLocaleString();
        },

        getOverallProgressPercentage() {
            const { questions_total, total_models, models_completed, questions_completed } = this.benchmarkProgress;
            if (!questions_total || !total_models) return 0;

            // Total questions across all models
            const totalQuestions = questions_total * total_models;
            // Questions completed so far (completed models + current model progress)
            const completedQuestions = (models_completed * questions_total) + questions_completed;

            return Math.round((completedQuestions / totalQuestions) * 100);
        },

        getModelProgressPercentage() {
            if (!this.benchmarkProgress.questions_total) return 0;
            return Math.round((this.benchmarkProgress.questions_completed / this.benchmarkProgress.questions_total) * 100);
        },

        getStatusBadgeClass(status) {
            const classes = {
                'idle': 'bg-gray-600',
                'running': 'bg-green-600',
                'completed': 'bg-blue-600',
                'failed': 'bg-red-600',
                'cancelled': 'bg-yellow-600'
            };
            return classes[status] || 'bg-gray-600';
        },

        viewBenchmarkResults(runId) {
            // Switch to individual evaluation page and select the run
            this.currentPage = 'individual';
            // The run should already be loaded, so it will appear in the dropdown
        },

        async viewJobLogs(jobId) {
            this.jobLogsLoading = true;
            this.showJobLogsModal = true;

            try {
                const response = await fetch(`/api/benchmark/jobs/${jobId}`);
                const data = await response.json();
                this.selectedJobForLogs = data;
            } catch (error) {
                console.error('Error loading job logs:', error);
                this.showToast('Failed to load job logs', 'error');
                this.showJobLogsModal = false;
            } finally {
                this.jobLogsLoading = false;
            }
        },

        closeJobLogsModal() {
            this.showJobLogsModal = false;
            this.selectedJobForLogs = null;
        },

        // ====================================================================
        // Toast Notifications
        // ====================================================================

        showToast(message, type = 'success') {
            const id = Date.now();
            this.toasts.push({ id, message, type });

            setTimeout(() => {
                this.toasts = this.toasts.filter(t => t.id !== id);
            }, 3000);
        },
    }
}).mount('#app');
