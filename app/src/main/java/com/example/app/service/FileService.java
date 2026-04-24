package com.example.app.service;

import com.example.app.data.FileEntity;
import com.example.app.data.Project;
import com.example.app.data.Task;
import com.example.app.repository.FileEntityRepository;
import com.example.app.repository.ProjectRepository;
import com.example.app.repository.TaskRepository;
import jakarta.persistence.EntityNotFoundException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.UUID;

@Service
public class FileService {

    private static final long MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

    private final FileEntityRepository fileRepository;
    private final ProjectRepository projectRepository;
    private final TaskRepository taskRepository;

    @Value("${file.upload-dir}")
    private String uploadDir;

    public FileService(FileEntityRepository fileRepository,
                       ProjectRepository projectRepository, TaskRepository taskRepository) {
        this.fileRepository = fileRepository;
        this.projectRepository = projectRepository;
        this.taskRepository = taskRepository;
    }

    public FileEntity uploadToProject(Long projectId, MultipartFile file) throws IOException {

        if (file.getSize() > MAX_FILE_SIZE) {
            throw new IllegalArgumentException("File too large");
        }
        Project project = projectRepository.findById(projectId)
                .orElseThrow();

        String fileName = UUID.randomUUID() + "_" + file.getOriginalFilename();
        Path path = Paths.get(uploadDir, fileName);

        Files.createDirectories(path.getParent());
        Files.write(path, file.getBytes());

        FileEntity entity = new FileEntity();
        entity.setName(file.getOriginalFilename());
        entity.setPath(path.toString());
        entity.setProject(project);

        return fileRepository.save(entity);
    }

    public FileEntity uploadToTask(Long taskId, MultipartFile file) throws IOException {
        if (file.getSize() > MAX_FILE_SIZE) {
            throw new IllegalArgumentException("File too large");
        }
        Task task = taskRepository.findById(taskId)
                .orElseThrow();

        String fileName = UUID.randomUUID() + "_" + file.getOriginalFilename();
        Path path = Paths.get(uploadDir, fileName);

        Files.createDirectories(path.getParent());
        Files.write(path, file.getBytes());

        FileEntity entity = new FileEntity();
        entity.setName(file.getOriginalFilename());
        entity.setPath(path.toString());
        entity.setTask(task);
        entity.setProject(null);

        return fileRepository.save(entity);
    }

    public List<FileEntity> getFilesByProject(Long projectId) {

        if (!projectRepository.existsById(projectId)) {
            throw new EntityNotFoundException("Project not found");
        }
        return fileRepository.findByProjectId(projectId);
    }

    public List<FileEntity> getFilesByTask(Long taskId) {

        if (!taskRepository.existsById(taskId)) {
            throw new EntityNotFoundException("Task not found");
        }
        return fileRepository.findByTaskId(taskId);
    }


    public byte[] downloadFile(Long fileId) throws Exception {

        FileEntity file = fileRepository.findById(fileId)
                .orElseThrow();

        return Files.readAllBytes(Paths.get(file.getPath()));
    }

    public void deleteFile(Long fileId) throws IOException {
        FileEntity file = fileRepository.findById(fileId)
                .orElseThrow();

        Path path = Paths.get(file.getPath());
        Files.deleteIfExists(path);
        fileRepository.delete(file);
    }
}