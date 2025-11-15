# CLAUDE.md - AI Assistant Guide for Agentic Hypergraph

## Project Overview

**agentic-hypergraph** is a project focused on implementing agentic systems using hypergraph data structures. This repository explores the intersection of:
- **Agentic AI Systems**: Autonomous agents that can reason, plan, and execute tasks
- **Hypergraph Structures**: Mathematical structures that generalize graphs by allowing edges to connect any number of vertices

## Repository Structure

```
agentic-hypergraph/
├── src/                    # Source code
│   ├── core/              # Core hypergraph implementations
│   ├── agents/            # Agent implementations
│   ├── algorithms/        # Graph algorithms and operations
│   ├── utils/             # Utility functions
│   └── types/             # TypeScript type definitions
├── tests/                 # Test files
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── examples/             # Example usage and demos
├── docs/                 # Documentation
├── scripts/              # Build and utility scripts
├── .github/              # GitHub workflows and templates
└── dist/                 # Compiled output (gitignored)
```

## Technology Stack

### Expected Technologies
- **Language**: TypeScript (preferred) or Python
- **Testing**: Jest/Vitest (TS) or pytest (Python)
- **Documentation**: Markdown, JSDoc/TSDoc
- **Version Control**: Git with conventional commits
- **Package Management**: npm/yarn (TS) or pip/poetry (Python)

## Development Workflows

### Git Workflow

1. **Branch Naming Convention**:
   - Feature branches: `feature/<description>`
   - Bug fixes: `fix/<issue-description>`
   - Documentation: `docs/<topic>`
   - AI assistant branches: `claude/<session-id>`

2. **Commit Message Format**:
   ```
   <type>(<scope>): <subject>

   <body>

   <footer>
   ```

   Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

   Examples:
   - `feat(hypergraph): add directed hyperedge support`
   - `fix(agent): resolve memory leak in task execution`
   - `docs(readme): update installation instructions`

3. **Pull Request Process**:
   - Create PR with descriptive title and summary
   - Include test plan and verification steps
   - Link related issues
   - Ensure CI/CD passes
   - Request review when ready

### Testing Requirements

- **Unit Tests**: Required for all core functionality
- **Integration Tests**: Required for agent interactions and complex workflows
- **Coverage Target**: Aim for >80% code coverage
- **Test Naming**: Use descriptive names that explain what is being tested
  ```typescript
  describe('HypergraphAgent', () => {
    it('should traverse hyperedges to find connected vertices', () => {
      // test implementation
    });
  });
  ```

### Code Quality Standards

1. **Type Safety**:
   - Use strict TypeScript settings (if TS)
   - Avoid `any` types; use proper type definitions
   - Export types for public APIs

2. **Documentation**:
   - All public functions must have JSDoc/docstring comments
   - Include parameter descriptions, return types, and examples
   - Document complex algorithms with inline comments

3. **Code Style**:
   - Use consistent formatting (Prettier/Black)
   - Follow language-specific style guides
   - Keep functions small and focused (single responsibility)
   - Use meaningful variable and function names

4. **Error Handling**:
   - Use proper error types and messages
   - Handle edge cases explicitly
   - Log errors appropriately
   - Provide helpful error messages

## Key Concepts and Conventions

### Hypergraph Terminology

- **Vertex/Node**: Basic unit in the hypergraph
- **Hyperedge**: Connection between one or more vertices (generalization of edge)
- **Incidence**: Relationship between vertices and hyperedges
- **Degree**: Number of hyperedges incident to a vertex
- **Order**: Number of vertices in a hyperedge

### Agent Architecture

Agents in this system should follow these principles:

1. **Autonomy**: Agents make independent decisions based on their state and goals
2. **Reactivity**: Agents respond to changes in their environment
3. **Proactivity**: Agents take initiative to achieve goals
4. **Social Ability**: Agents can interact with other agents through the hypergraph

### Code Organization Patterns

1. **Hypergraph Core**:
   ```typescript
   class Hypergraph {
     vertices: Set<Vertex>
     hyperedges: Set<Hyperedge>

     addVertex(vertex: Vertex): void
     addHyperedge(hyperedge: Hyperedge): void
     getNeighbors(vertex: Vertex): Set<Vertex>
   }
   ```

2. **Agent Interface**:
   ```typescript
   interface Agent {
     id: string
     state: AgentState
     perceive(environment: Hypergraph): Perception
     decide(perception: Perception): Action
     act(action: Action): void
   }
   ```

3. **Algorithm Pattern**:
   ```typescript
   function algorithmName(
     graph: Hypergraph,
     options?: AlgorithmOptions
   ): Result {
     // Algorithm implementation
     // Return structured result
   }
   ```

## AI Assistant Guidelines

### When Working on This Codebase

1. **Understanding Context**:
   - Read relevant documentation before making changes
   - Understand the hypergraph structure being modified
   - Consider impact on agent behaviors

2. **Making Changes**:
   - Always write tests for new functionality
   - Update documentation when changing APIs
   - Maintain backward compatibility when possible
   - Use the TodoWrite tool to track multi-step tasks

3. **Code Review Checklist**:
   - [ ] Types are properly defined
   - [ ] Functions have documentation
   - [ ] Tests are written and passing
   - [ ] No security vulnerabilities introduced
   - [ ] Performance implications considered
   - [ ] Edge cases handled

4. **Common Patterns to Follow**:
   - Immutability: Prefer immutable data structures
   - Pure Functions: Avoid side effects when possible
   - Composition: Build complex behavior from simple parts
   - Dependency Injection: Pass dependencies explicitly

5. **Performance Considerations**:
   - Be mindful of time complexity for graph operations
   - Use appropriate data structures (Map, Set, etc.)
   - Consider memory usage for large hypergraphs
   - Profile before optimizing

### File Creation Guidelines

- **Never create unnecessary files**: Only create files that are essential for functionality
- **Prefer editing**: Always prefer editing existing files over creating new ones
- **Documentation**: Create documentation only when explicitly requested
- **Examples**: Create example files only when they demonstrate important concepts

### Security Best Practices

- **Input Validation**: Validate all external inputs
- **Sanitization**: Sanitize data before processing
- **Dependency Security**: Keep dependencies updated
- **No Secrets**: Never commit secrets or credentials
- **OWASP Awareness**: Prevent injection attacks, XSS, etc.

## Common Operations

### Adding a New Hypergraph Algorithm

1. Create algorithm file in `src/algorithms/`
2. Implement with proper types and error handling
3. Write unit tests in `tests/unit/algorithms/`
4. Add integration test if needed
5. Document in `docs/algorithms.md`
6. Export from `src/algorithms/index.ts`

### Adding a New Agent Type

1. Create agent class in `src/agents/`
2. Implement Agent interface
3. Define agent-specific state and behavior
4. Write tests for agent logic
5. Create example usage in `examples/`
6. Document agent capabilities

### Running Tests

```bash
# Run all tests
npm test

# Run specific test file
npm test -- path/to/test

# Run with coverage
npm test -- --coverage

# Watch mode
npm test -- --watch
```

### Building the Project

```bash
# Development build
npm run build:dev

# Production build
npm run build

# Clean build
npm run clean && npm run build
```

## Debugging and Troubleshooting

### Common Issues

1. **Type Errors**: Check type definitions in `src/types/`
2. **Graph Traversal Issues**: Verify hyperedge connections
3. **Agent Communication**: Check hypergraph structure for agent connectivity
4. **Performance**: Profile with appropriate tools, check algorithm complexity

### Debugging Tools

- Use debugger statements and breakpoints
- Log hypergraph state at critical points
- Visualize hypergraph structure when debugging complex issues
- Use type checking to catch errors early

## Resources

### Hypergraph Theory
- Understanding hypergraph data structures
- Common hypergraph algorithms
- Applications in multi-relational data

### Agentic Systems
- Agent architectures and patterns
- Multi-agent coordination
- Decision-making algorithms

## Contributing

When contributing to this project:

1. Follow all conventions outlined in this document
2. Write clear, documented, tested code
3. Consider the impact on the overall system
4. Communicate changes clearly in commits and PRs
5. Be open to feedback and iteration

## Version History

- **v0.1.0**: Initial project setup and CLAUDE.md creation

---

**Last Updated**: 2025-11-15
**Maintained by**: AI assistants working on this project

For questions or clarifications, refer to additional documentation in the `docs/` directory or update this file as the project evolves.
