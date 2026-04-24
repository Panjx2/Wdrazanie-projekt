package com.project.service;

import com.project.model.Project;
import java.util.List;

public interface ProjectService {
    List<Project> getAllProjects();

    Project getProjectById(Long id);

    Project createProject(Project project);

    Project updateProject(Long id, Project project);

    void deleteProject(Long id);

    Project addUserToProject(Long projectId, Long userId);

    Project removeUserFromProject(Long projectId, Long userId);
}