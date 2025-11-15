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
                { id: 'authors-choice', name: "Author's Choice", icon: 'star' }
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
            selectedQuestionForRating: null,
            currentRating: { score: 50, comment: '' },

            // Author's Choice Data
            authorsChoiceRankings: [],
            authorsChoiceLoading: false,

            // Config Editor Data
            configContent: '',
            configErrors: [],
            configSaving: false,
            configBackups: [],
            configSection: 'judge', // Current config section: judge, fixer, models, filtering, performance, other
            showYamlCodeModal: false,

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
            jsonErrors: {}
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

        visualConfig: {
            handler() {
                // Sync visual config to YAML in real-time
                this.syncVisualToYaml();
            },
            deep: true
        },

        judgeModelConfig: {
            handler() {
                this.syncJudgeFixerToVisualConfig();
            },
            deep: true
        },

        fixerModelConfig: {
            handler() {
                this.syncJudgeFixerToVisualConfig();
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
            try {
                const response = await fetch('/api/config');
                const data = await response.json();
                this.configContent = data.content || '';

                // Parse YAML for visual editor
                try {
                    const parsed = jsyaml.load(this.configContent);
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
                    this.showToast('Error parsing config YAML', 'error');
                }

                // Initialize CodeMirror if not already initialized
                this.$nextTick(() => {
                    this.initializeCodeMirror();
                });

                // Load backups
                await this.loadConfigBackups();

            } catch (error) {
                console.error('Error loading config:', error);
                this.showToast('Failed to load config', 'error');
            }
        },

        initializeCodeMirror() {
            const textarea = document.getElementById('config-editor');
            if (!textarea || !window.CodeMirror) return;

            // Check if already initialized
            if (textarea.nextSibling && textarea.nextSibling.classList && textarea.nextSibling.classList.contains('CodeMirror')) {
                return;
            }

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

            editor.on('change', (cm) => {
                this.configContent = cm.getValue();
            });

            // Store editor instance
            this.configEditor = editor;
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
            // Validate first
            await this.validateConfig();

            if (this.configErrors.length > 0) {
                this.showToast('Please fix validation errors before saving', 'error');
                return;
            }

            this.configSaving = true;
            try {
                const response = await fetch('/api/config/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        yaml_content: this.configContent
                    })
                });

                const data = await response.json();
                this.showToast(data.message || 'Config saved', 'success');

                // Reload backups
                await this.loadConfigBackups();

            } catch (error) {
                console.error('Error saving config:', error);
                this.showToast('Failed to save config', 'error');
            } finally {
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

        // ====================================================================
        // Visual Config Editor Methods
        // ====================================================================

        syncVisualToYaml() {
            // Sync judge and fixer configs into model_configs
            this.syncJudgeFixerToVisualConfig();

            // Convert visualConfig to YAML
            try {
                const yamlContent = jsyaml.dump(this.visualConfig, {
                    indent: 2,
                    lineWidth: 120,
                    noRefs: true,
                    sortKeys: false
                });
                this.configContent = yamlContent;

                // Update CodeMirror if it exists
                if (this.configEditor) {
                    this.configEditor.setValue(yamlContent);
                }
            } catch (error) {
                console.error('Error converting to YAML:', error);
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

        validateJSON(type) {
            // Validate JSON for judge or fixer
            let jsonText = type === 'judge' ? this.judgeModelOptionsJSON : this.fixerModelOptionsJSON;

            if (!jsonText.trim()) {
                this.jsonErrors[type] = '';
                if (type === 'judge') {
                    this.judgeModelConfig.options = {};
                } else {
                    this.fixerModelConfig.options = {};
                }
                return;
            }

            // Smart JSON: Auto-add braces if missing
            jsonText = jsonText.trim();
            if (!jsonText.startsWith('{')) {
                jsonText = '{' + jsonText + '}';
            }

            try {
                const parsed = JSON.parse(jsonText);
                this.jsonErrors[type] = '';
                if (type === 'judge') {
                    this.judgeModelConfig.options = parsed;
                } else {
                    this.fixerModelConfig.options = parsed;
                }
            } catch (error) {
                this.jsonErrors[type] = error.message;
            }
        },

        validateModelOptionsJSON(modelId) {
            let jsonText = this.editingModelOptionsJSON[modelId];

            if (!jsonText || !jsonText.trim()) {
                this.jsonErrors['model_' + modelId] = '';
                this.editingModelConfigs[modelId].options = {};
                return;
            }

            // Smart JSON: Auto-add braces if missing
            jsonText = jsonText.trim();
            if (!jsonText.startsWith('{')) {
                jsonText = '{' + jsonText + '}';
            }

            try {
                const parsed = JSON.parse(jsonText);
                this.jsonErrors['model_' + modelId] = '';
                this.editingModelConfigs[modelId].options = parsed;
            } catch (error) {
                this.jsonErrors['model_' + modelId] = error.message;
            }
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
                    try {
                        // Smart JSON: Auto-add braces if missing
                        let jsonText = this.newModel.optionsJSON.trim();
                        if (!jsonText.startsWith('{')) {
                            jsonText = '{' + jsonText + '}';
                        }
                        config.options = JSON.parse(jsonText);
                    } catch (error) {
                        this.showToast('Invalid JSON in options', 'error');
                        return;
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
                    JSON.stringify(config.options, null, 2) : '';
            }

            // Re-render icons
            this.$nextTick(() => {
                if (window.lucide) {
                    lucide.createIcons();
                }
            });
        },

        saveModelConfig(modelId) {
            // Validate JSON first
            this.validateModelOptionsJSON(modelId);
            if (this.jsonErrors['model_' + modelId]) {
                this.showToast('Fix JSON errors before saving', 'error');
                return;
            }

            // Save config
            const config = this.editingModelConfigs[modelId];
            if (config.system_instruction || Object.keys(config.options).length > 0) {
                this.visualConfig.model_configs[modelId] = { ...config };
            } else {
                delete this.visualConfig.model_configs[modelId];
            }

            this.showToast(`Model ${modelId} configuration saved`, 'success');
            this.toggleModelExpand(modelId);
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
