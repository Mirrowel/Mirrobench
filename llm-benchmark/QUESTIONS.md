# LLM Benchmark Questions Catalog

This document provides a comprehensive overview of all questions in the LLM benchmark, organized by category with links to detailed documentation.

## Overview

The benchmark contains **25 questions** across **7 categories**, testing a wide range of programming skills from CLI tools to creative coding. Questions are designed to create complete, feature-rich applications rather than minimal implementations.

### Quick Stats
- **Total Questions:** 25
- **Categories:** 7
- **Difficulty Range:** Medium to Very Hard
- **Evaluation Methods:** Code Execution, LLM Judge, Tool Validation
- **Languages:** Python, JavaScript/Node.js, Rust, HTML/CSS/JS

---

## Question Categories

### 1. CLI Tools - Productivity Tools (5 questions)
**Documentation:** [docs/questions_cli_tools.md](docs/questions_cli_tools.md)

Focus on building practical command-line applications with emphasis on usability and features.

| Question | ID | Difficulty | Language | Focus |
|----------|----|----|----|----|
| Terminal Text Editor | `cli_terminal_text_editor` | Very Hard | Python | Text editing, curses UI |
| File Organizer | `cli_file_organizer_python` | Medium | Python | File management, safety |
| Weather Dashboard | `cli_weather_dashboard_rust` | Hard | Rust | Terminal UI, data display |
| Markdown to HTML Converter | `cli_markdown_to_html_node` | Hard | Node.js | Text processing, parsing |
| System Monitor | `cli_system_monitor_python` | Hard | Python | Real-time monitoring |

---

### 2. Games (10 questions)
**Documentation:** [docs/questions_games.md](docs/questions_games.md)

Comprehensive game development covering arcade, puzzle, 3D, and multiplayer games.

#### Python Games (6 questions)
| Question | ID | Difficulty | Type |
|----------|----|----|----|
| Snake Game | `game_snake_pygame` | Medium | Arcade |
| Space Shooter | `game_space_shooter_pygame` | Hard | Shooter |
| Tetris | `game_tetris_python` | Very Hard | Puzzle |
| Voxel World | `game_voxel_world_python` | Very Hard | 3D |
| First-Person Shooter | `game_fps_shooter_python` | Very Hard | 3D |
| Roguelike Dungeon | `game_roguelike_dungeon` | Very Hard | Roguelike |

#### Web Games (4 questions)
| Question | ID | Difficulty | Type |
|----------|----|----|----|
| 2048 Game | `game_2048_html` | Hard | Puzzle |
| 3D Historical Scene | `viz_3d_historical_scene` | Very Hard | 3D Visualization |
| 3D Bird Forest | `viz_3d_bird_forest` | Very Hard | 3D Animation |
| Multiplayer Game | `game_multiplayer_realtime` | Very Hard | Multiplayer |

---

### 3. Web Applications (4 questions)
**Documentation:** [docs/questions_web_apps.md](docs/questions_web_apps.md)

Interactive single-page web applications with emphasis on UX and functionality.

| Question | ID | Difficulty | Focus |
|----------|----|----|----|
| Drawing App | `webapp_drawing_app` | Hard | Canvas, creative tools |
| Quiz App | `webapp_quiz_app` | Medium | Education, interactivity |
| Expense Tracker | `webapp_expense_tracker` | Very Hard | CRUD, data visualization |
| Markdown Editor | `webapp_markdown_editor` | Very Hard | Text processing, preview |

---

### 4. Visualizations (4 questions)
**Documentation:** [docs/questions_visualizations.md](docs/questions_visualizations.md)

Graphics and data visualization projects using Canvas, SVG, and web technologies.

| Question | ID | Difficulty | Technology |
|----------|----|----|----|
| SVG Analog Clock | `viz_svg_clock` | Medium | SVG, animation |
| Fractal Tree Generator | `viz_fractal_tree` | Hard | Canvas, recursion |
| Particle System | `viz_particle_system` | Hard | Canvas, physics |
| Data Dashboard | `viz_data_dashboard` | Very Hard | Charts, interactivity |

---

### 5. Simulations (5 questions)
**Documentation:** [docs/questions_simulations.md](docs/questions_simulations.md)

Complex physics and nature simulations requiring mathematical accuracy.

| Question | ID | Difficulty | Domain |
|----------|----|----|----|
| Solar System | `sim_solar_system` | Very Hard | Astronomy |
| Double Pendulum | `sim_double_pendulum` | Very Hard | Chaos physics |
| Boids Flocking | `sim_boids_flocking` | Very Hard | Artificial life |
| Fluid Dynamics | `sim_fluid_dynamics` | Very Hard | Computational physics |
| 2D Physics Engine | `engine_physics_2d` | Very Hard | Physics engine |

---

### 6. Creative Coding (6 questions)
**Documentation:** [docs/questions_creative_coding.md](docs/questions_creative_coding.md)

Advanced creative applications combining art, mathematics, and technology.

| Question | ID | Difficulty | Domain |
|----------|----|----|----|
| Mandelbrot Explorer | `creative_mandelbrot_set` | Very Hard | Fractals |
| Music Visualizer | `creative_music_visualizer` | Very Hard | Audio visualization |
| Ray Tracer | `creative_ray_tracer` | Very Hard | 3D graphics |
| Conway's Game of Life | `creative_conway_life` | Hard | Cellular automata |
| Path Tracer | `graphics_path_tracer` | Very Hard | Global illumination |
| Music Tracker | `creative_music_tracker` | Very Hard | Audio synthesis |

---

### 7. Todo App (1 question)
**Documentation:** [docs/questions_todo_app.md](docs/questions_todo_app.md)

Multi-file web application testing file organization and frontend skills.

| Question | ID | Difficulty | Focus |
|----------|----|----|----|
| Todo List App | `web_app_todo_multi_file` | Medium | Multi-file, CRUD |

---

## Evaluation Criteria

### Evaluation Methods

1. **Code Execution** (10 questions)
   - Automated testing of functionality
   - Used for questions with clear correct/incorrect outcomes
   - Examples: Games, basic tools, simple apps

2. **LLM Judge** (14 questions)
   - Qualitative assessment by AI
   - Used for creative, visual, or complex implementations
   - Examples: 3D graphics, simulations, creative coding

3. **Tool Validation** (1 question)
   - Specialized validation for specific tool requirements
   - Used for questions requiring specific tool usage

### Common Evaluation Factors

- **Functionality:** Does the application work as specified?
- **Feature Completeness:** Are expected features implemented?
- **Code Quality:** Is the code well-structured and maintainable?
- **User Experience:** Is the interface intuitive and polished?
- **Creativity:** For creative questions, artistic and innovative elements
- **Technical Accuracy:** For simulations, mathematical and scientific correctness

---

## Difficulty Distribution

- **Medium:** 4 questions (16%)
- **Hard:** 7 questions (28%)
- **Very Hard:** 14 questions (56%)

Most questions are challenging and require substantial implementation effort, reflecting real-world development complexity.

---

## Technology Stack

### Languages
- **Python:** 11 questions (44%)
- **JavaScript/Node.js:** 9 questions (36%)
- **HTML/CSS/JS:** 8 questions (32%)
- **Rust:** 1 question (4%)

### Key Technologies
- **Pygame:** Python game development
- **Canvas API:** Web graphics and animations
- **WebGL:** 3D browser graphics
- **Web Audio API:** Audio processing and visualization
- **Curses:** Terminal UI applications
- **LocalStorage:** Browser data persistence

---

## Question Evolution

**Note:** Questions are subject to frequent updates and improvements. The benchmark is actively maintained to:

- Add new question categories and types
- Refine existing prompts for clarity
- Update evaluation criteria
- Fix issues discovered during testing
- Adjust difficulty levels based on performance data

Always refer to the latest documentation files for the most current question specifications.

---

## Usage Guidelines

### For Benchmarking
- Use questions appropriate to the model's expected capabilities
- Consider difficulty when selecting questions for evaluation
- Allow adequate time for complex implementations
- Review detailed documentation for specific requirements

### For Development Practice
- Start with medium difficulty questions
- Progress to harder questions as skills improve
- Focus on complete, polished implementations
- Study the evaluation criteria to understand expectations

---

## File Structure

Questions are organized in the following directory structure:

```
questions/
├── cli_tools/
│   └── productivity_tools.json
├── creative_coding/
│   └── generative_and_interactive.json
├── games/
│   ├── python_games.json
│   └── web_games.json
├── simulations/
│   └── physics_and_nature.json
├── visualizations/
│   └── graphics_and_art.json
├── web_apps/
│   ├── interactive_apps.json
│   └── todo_app_multi_file.json
```

Each JSON file contains the complete question definitions, prompts, system prompts, and evaluation criteria for that category.

---

## Contributing

When adding new questions or modifying existing ones:

1. Follow the established format and structure
2. Provide clear, comprehensive prompts
3. Include appropriate evaluation criteria
4. Update this catalog document
5. Test questions thoroughly before inclusion
6. Document any special requirements or considerations

---

*This catalog serves as a reference guide. For detailed specifications, prompts, and evaluation criteria, please refer to the individual category documentation files linked above.*