package com.example.app;

import com.example.app.data.FileEntity;
import com.example.app.data.Project;
import com.example.app.data.Task;
import com.example.app.repository.FileEntityRepository;
import com.example.app.repository.ProjectRepository;
import com.example.app.repository.TaskRepository;
import com.example.app.service.FileService;
import jakarta.persistence.EntityNotFoundException;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.springframework.mock.web.MockMultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(org.mockito.junit.jupiter.MockitoExtension.class)
class FileServiceTest {

    @Mock
    private FileEntityRepository fileRepository;

    @Mock
    private ProjectRepository projectRepository;

    @Mock
    private TaskRepository taskRepository;

    @InjectMocks
    private FileService fileService;

    private final String TEST_DIR = "test-uploads";

    @BeforeEach
    void setup() throws IOException {
        Path path = Path.of(TEST_DIR);

        if (!Files.exists(path)) {
            Files.createDirectories(path);
        }

        org.springframework.test.util.ReflectionTestUtils
                .setField(fileService, "uploadDir", TEST_DIR);
    }

    @AfterEach
    void cleanup() throws IOException {
        Files.walk(Path.of(TEST_DIR))
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(java.io.File::delete);
    }

    @Test
    void shouldUploadFileToProject() throws IOException {

        Project project = new Project();
        when(projectRepository.findById(1L)).thenReturn(Optional.of(project));
        when(fileRepository.save(any())).thenAnswer(i -> i.getArgument(0));

        MockMultipartFile file =
                new MockMultipartFile("file", "test.txt", "text/plain", "data".getBytes());

        FileEntity result = fileService.uploadToProject(1L, file);

        assertEquals("test.txt", result.getName());
        assertNotNull(result.getPath());
        assertEquals(project, result.getProject());
    }

    @Test
    void shouldUploadFileToTask() throws IOException {

        Task task = new Task();
        when(taskRepository.findById(1L)).thenReturn(Optional.of(task));
        when(fileRepository.save(any())).thenAnswer(i -> i.getArgument(0));

        MockMultipartFile file =
                new MockMultipartFile("file", "test.txt", "text/plain", "data".getBytes());

        FileEntity result = fileService.uploadToTask(1L, file);

        assertEquals("test.txt", result.getName());
        assertEquals(task, result.getTask());
        assertNull(result.getProject());
    }

    @Test
    void shouldGetFilesByProject() {

        when(projectRepository.existsById(1L)).thenReturn(true);
        when(fileRepository.findByProjectId(1L))
                .thenReturn(List.of(new FileEntity()));

        List<FileEntity> result = fileService.getFilesByProject(1L);

        assertEquals(1, result.size());
    }

    @Test
    void shouldThrowWhenProjectNotFound() {

        when(projectRepository.existsById(1L)).thenReturn(false);

        assertThrows(EntityNotFoundException.class,
                () -> fileService.getFilesByProject(1L));
    }

    @Test
    void shouldGetFilesByTask() {

        when(taskRepository.existsById(1L)).thenReturn(true);
        when(fileRepository.findByTaskId(1L))
                .thenReturn(List.of(new FileEntity()));

        List<FileEntity> result = fileService.getFilesByTask(1L);

        assertEquals(1, result.size());
    }

    @Test
    void shouldThrowWhenTaskNotFound() {

        when(taskRepository.existsById(1L)).thenReturn(false);

        assertThrows(EntityNotFoundException.class,
                () -> fileService.getFilesByTask(1L));
    }

    @Test
    void shouldDownloadFile() throws Exception {

        Path tempFile = Files.createTempFile("test", ".txt");
        Files.write(tempFile, "data".getBytes());

        FileEntity entity = new FileEntity();
        entity.setPath(tempFile.toString());

        when(fileRepository.findById(1L)).thenReturn(Optional.of(entity));

        byte[] result = fileService.downloadFile(1L);

        assertArrayEquals("data".getBytes(), result);
    }


    @Test
    void shouldDeleteFile() throws IOException {

        Path tempFile = Files.createTempFile("test", ".txt");

        FileEntity entity = new FileEntity();
        entity.setPath(tempFile.toString());

        when(fileRepository.findById(1L)).thenReturn(Optional.of(entity));

        fileService.deleteFile(1L);

        assertFalse(Files.exists(tempFile));
        verify(fileRepository).delete(entity);
    }
}
