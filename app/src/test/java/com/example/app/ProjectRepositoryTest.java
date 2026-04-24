package com.example.app;

import com.example.app.data.Project;
import com.example.app.repository.ProjectRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.data.jpa.test.autoconfigure.DataJpaTest;
import static org.junit.jupiter.api.Assertions.*;

@DataJpaTest
class ProjectRepositoryTest {

    @Autowired
    ProjectRepository projectRepository;

    @Test
    void shouldSaveProject() {

        Project p = new Project();
        p.setName("Test Project");

        projectRepository.save(p);

        assertFalse(projectRepository.findAll().isEmpty());
    }
}