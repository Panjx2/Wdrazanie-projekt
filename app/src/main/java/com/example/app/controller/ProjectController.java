package com.example.app.controller;

import com.example.app.dto.ProjectDto;
import com.example.app.mapper.DtoMapper;
import com.example.app.service.ProjectService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/projects")
public class ProjectController {

    private final ProjectService projectService;
    private final DtoMapper dtoMapper;

    public ProjectController(ProjectService projectService, DtoMapper dtoMapper) {
        this.projectService = projectService;
        this.dtoMapper = dtoMapper;
    }

    @PostMapping
    public ProjectDto createProject(@RequestBody ProjectDto projectDto) {
        return dtoMapper.toDto(projectService.createProject(dtoMapper.toEntity(projectDto)));
    }

    @GetMapping
    public List<ProjectDto> getAllProjects() {
        return projectService.getAllProjects().stream()
                .map(dtoMapper::toDto)
                .toList();
    }

    @GetMapping("/{id}")
    public ProjectDto getProjectById(@PathVariable Long id) {
        return dtoMapper.toDto(projectService.getProjectById(id));
    }

    @PutMapping("/{id}")
    public ProjectDto updateProject(@PathVariable Long id, @RequestBody ProjectDto projectDto) {
        return dtoMapper.toDto(projectService.updateProject(id, dtoMapper.toEntity(projectDto)));
    }

    @DeleteMapping("/{id}")
    public void deleteProject(@PathVariable Long id) {
        projectService.deleteProject(id);
    }

    @PostMapping("/{projectId}/users/{userId}")
    public ProjectDto addUserToProject(@PathVariable Long projectId,
                                       @PathVariable Long userId) {
        return dtoMapper.toDto(projectService.addUserToProject(projectId, userId));
    }

    @DeleteMapping("/{projectId}/users/{userId}")
    public ProjectDto removeUserFromProject(@PathVariable Long projectId,
                                            @PathVariable Long userId) {
        return dtoMapper.toDto(projectService.removeUserFromProject(projectId, userId));
    }
}
