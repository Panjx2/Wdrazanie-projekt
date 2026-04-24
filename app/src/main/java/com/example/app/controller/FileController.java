package com.example.app.controller;

import com.example.app.data.FileEntity;
import com.example.app.service.FileService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api/files")
public class FileController {

    private final FileService fileService;

    public FileController(FileService fileService) {
        this.fileService = fileService;
    }

    @PostMapping("/project/{projectId}")
    public FileEntity uploadToProject(
            @PathVariable Long projectId,
            @RequestParam("file") MultipartFile file) throws IOException {

        return fileService.uploadToProject(projectId, file);
    }

    @PostMapping("/task/{taskId}")
    public FileEntity uploadToTask(
            @PathVariable Long taskId,
            @RequestParam("file") MultipartFile file) throws IOException {

        return fileService.uploadToTask(taskId, file);
    }

    @GetMapping("/project/{projectId}")
    public List<FileEntity> getProjectFiles(@PathVariable Long projectId) {
        return fileService.getFilesByProject(projectId);
    }


    @GetMapping("/task/{taskId}")
    public List<FileEntity> getTaskFiles(@PathVariable Long taskId) {
        return fileService.getFilesByTask(taskId);
    }

    @DeleteMapping("/{fileId}")
    public void deleteFile(@PathVariable Long fileId) throws IOException {
        fileService.deleteFile(fileId);
    }

    @GetMapping("/{fileId}")
    public ResponseEntity<byte[]> downloadFile(@PathVariable Long fileId) throws Exception {

        byte[] data = fileService.downloadFile(fileId);

        return ResponseEntity.ok()
                .header("Content-Disposition", "attachment; filename=file")
                .body(data);
    }

}

