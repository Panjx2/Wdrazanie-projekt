package com.example.app.repository;

import com.example.app.data.FileEntity;
import com.example.app.data.Project;
import com.example.app.data.Task;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface FileEntityRepository extends JpaRepository<FileEntity, Long> {
    List<FileEntity> findByProject(Project project);
    List<FileEntity> findByTask(Task task);

    List<FileEntity> findByProjectId(Long projectId);

    List<FileEntity> findByTaskId(Long taskId);
}