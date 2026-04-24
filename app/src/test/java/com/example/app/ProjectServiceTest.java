package com.example.app;

import com.example.app.data.Project;
import com.example.app.data.User;
import com.example.app.repository.ProjectRepository;
import com.example.app.repository.UserRepository;
import com.example.app.service.ProjectService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.springframework.data.domain.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(org.mockito.junit.jupiter.MockitoExtension.class)
class ProjectServiceTest {

    @Mock
    private ProjectRepository projectRepository;

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private ProjectService projectService;

    @Test
    void shouldCreateProject() {
        Project project = new Project();
        project.setName("Test");

        when(projectRepository.save(project)).thenReturn(project);

        Project result = projectService.createProject(project);

        assertEquals("Test", result.getName());
        verify(projectRepository).save(project);
    }

    @Test
    void shouldGetProjectById() {
        Project project = new Project();
        project.setId(1L);

        when(projectRepository.findById(1L)).thenReturn(Optional.of(project));

        Project result = projectService.getProjectById(1L);

        assertEquals(1L, result.getId());
    }

    @Test
    void shouldUpdateProject() {
        Project existing = new Project();
        existing.setId(1L);
        existing.setName("Old");

        Project updated = new Project();
        updated.setName("New");

        when(projectRepository.findById(1L)).thenReturn(Optional.of(existing));
        when(projectRepository.save(any())).thenReturn(existing);

        Project result = projectService.updateProject(1L, updated);

        assertEquals("New", result.getName());
    }

    @Test
    void shouldAddUserToProject() {
        Project project = new Project();
        project.setUsers(new HashSet<>());

        User user = new User();

        when(projectRepository.findById(1L)).thenReturn(Optional.of(project));
        when(userRepository.findById(2L)).thenReturn(Optional.of(user));
        when(projectRepository.save(project)).thenReturn(project);

        Project result = projectService.addUserToProject(1L, 2L);

        assertTrue(result.getUsers().contains(user));
    }

    @Test
    void shouldFilterProjectsByName() {
        Pageable pageable = PageRequest.of(0, 10);
        Page<Project> page = new PageImpl<>(List.of(new Project()));

        when(projectRepository.findByNameContainingIgnoreCase("test", pageable))
                .thenReturn(page);

        Page<Project> result = projectService.getProjects("test", pageable);

        assertEquals(1, result.getTotalElements());
    }
}